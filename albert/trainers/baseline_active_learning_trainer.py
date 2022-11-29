# pylint: disable=logging-too-many-args
from __future__ import absolute_import

import datetime
import logging
import os
import time
from collections import MutableMapping
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from overrides import overrides

import allennlp.nn.util as nn_util
from albert.acquisition_functions.base_acquisition_function import (
    BaseAcquisitionFunction,
)
from albert.utils.trainer_utils import create_optimizer
from allennlp.common import Params
from allennlp.common.checks import parse_cuda_device
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import dump_metrics, gpu_memory_mb, peak_memory_mb
from allennlp.data import DataIterator, Instance
from allennlp.models import Model
from allennlp.training import TrainerBase
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.tensorboard_writer import TensorboardWriter

logger = logging.getLogger(__name__)


@TrainerBase.register("baseline_active_learning")
class BaselineActiveLearningTrainer(TrainerBase):
    def __init__(
        self,
        model: Model,
        optimizer_params: Params,
        train_data: Iterable[Instance],
        validation_data: Iterable[Instance],
        unlabeled_data: Iterable[Instance],
        validation_metric: str,
        patience: int,
        num_iterations: int,
        num_epochs: int,
        batcher: DataIterator,
        acquisition_function: BaseAcquisitionFunction,
        serialization_dir: str,
        learning_rate_scheduler_params: Optional[Params] = None,
        grad_norm: Optional[float] = None,
        test_batcher: Optional[DataIterator] = None,
        test_data: Optional[Iterable[Instance]] = None,
        cuda_device: int = -1,
    ) -> None:
        super(BaselineActiveLearningTrainer, self).__init__(
            serialization_dir, cuda_device
        )
        self.model = self._move_to_gpu(model)
        if self.cuda_device > -1:
            logger.info("Running on the GPU")
        else:
            logger.info("Running on the CPU")
        self.optimizer = create_optimizer(
            optimizer_params=optimizer_params, model=self.model
        )

        self.train_data = train_data
        self.validation_data = validation_data
        self.unlabeled_data = unlabeled_data
        self.test_data = test_data

        self.acquisition_function = acquisition_function

        self.num_iterations = num_iterations
        self.num_epochs = num_epochs

        self.batcher = batcher
        self.test_batcher = test_batcher or batcher

        self.patience = patience if patience >= 0.0 else None
        self.validation_metric = validation_metric
        self._grad_norm = grad_norm

        self._batch_num_total = 0
        self._learning_rate_scheduler = None
        if learning_rate_scheduler_params:
            learning_rate_scheduler = LearningRateScheduler.from_params(
                optimizer=self.optimizer, params=learning_rate_scheduler_params
            )
            self._learning_rate_scheduler = learning_rate_scheduler

        # Tensorboard related items
        self._tensorboard = TensorboardWriter(
            get_batch_num_total=lambda: self._batch_num_total,
            serialization_dir=serialization_dir,
            summary_interval=100,
            histogram_interval=None,
            should_log_parameter_statistics=False,
            should_log_learning_rate=False,
        )

    def validation_loss(
        self, instance_list: Optional[List[Instance]] = None, test: bool = False
    ):
        logger.info("Validating")
        with torch.no_grad():
            self.model.eval()

            instance_list = instance_list or self.validation_data
            if test:
                raw_val_generator = self.test_batcher(
                    instance_list, num_epochs=1, shuffle=False
                )
                num_validation_batches = self.test_batcher.get_num_batches(
                    instance_list
                )
            else:
                raw_val_generator = self.batcher(
                    instance_list, num_epochs=1, shuffle=False
                )
                num_validation_batches = self.batcher.get_num_batches(instance_list)

            batches_this_epoch = 0
            val_loss = 0
            loss_count = 0
            tqdm_handle = Tqdm.tqdm(
                range(num_validation_batches), total=num_validation_batches
            )
            for _ in tqdm_handle:
                batches_this_epoch += 1
                tensor_dict = self.sample_from_iterator(raw_val_generator)
                loss = self.model(**tensor_dict)["loss"]
                if loss is not None:
                    # You shouldn't necessarily have to compute a loss for validation, so we allow for
                    # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                    # currently only used as the divisor for the loss function, so we can safely only
                    # count those batches for which we actually have a loss.  If this variable ever
                    # gets used for something else, we might need to change things around a bit.
                    loss_count += 1
                    val_loss += loss.item()

                # Update the description with the latest metrics
                val_metrics = training_util.get_metrics(
                    self.model, val_loss, loss_count
                )
                description = training_util.description_from_metrics(val_metrics)
                tqdm_handle.set_description(description, refresh=False)

        return val_loss, loss_count

    def sample_from_iterator(self, iterator) -> Dict[str, torch.Tensor]:
        tensor_dict = next(iterator)
        tensor_dict = nn_util.move_to_device(tensor_dict, self.cuda_device)
        return tensor_dict

    def rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self.model, self._grad_norm)

    def train_one_epoch(self, epoch: int) -> Tuple[Dict[str, Any], int]:

        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")
        logger.info(f"Epoch {epoch + 1} / {self.num_epochs}")

        self.model.train()

        raw_train_iterator = self.batcher(
            self.train_data, num_epochs=None, shuffle=True
        )
        num_batches = self.batcher.get_num_batches(self.train_data)
        total_loss = 0.0
        batches_this_epoch = 0.0
        metrics = {}
        tqdm_handle = Tqdm.tqdm(range(num_batches), total=num_batches)
        for bix in tqdm_handle:
            batches_this_epoch += 1
            self._batch_num_total += 1
            self.optimizer.zero_grad()
            # Dict[str, torch.Tensor]
            tensor_dict = self.sample_from_iterator(raw_train_iterator)
            loss = self.model(**tensor_dict)["loss"]
            total_loss += loss.item()
            loss.backward()
            self.rescale_gradients()

            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(self._batch_num_total)

            self.optimizer.step()
            metrics = training_util.get_metrics(
                self.model, total_loss, batches_this_epoch
            )
            description = training_util.description_from_metrics(metrics)
            tqdm_handle.set_description(description)

        metrics = training_util.get_metrics(
            self.model, total_loss, batches_this_epoch, reset=True
        )

        return metrics, num_batches

    def train_one_iteration(self, iteration: int) -> Dict[str, float]:
        metric_tracker = MetricTracker(self.patience, self.validation_metric)
        validation_metric = self.validation_metric[1:]
        training_start_time = time.time()
        metrics = {}

        serialization_dir = os.path.join(
            self._serialization_dir, f"iteration-{iteration}"
        )
        os.makedirs(serialization_dir, exist_ok=True)
        checkpointer = Checkpointer(serialization_dir, None, 1)

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            training_metrics, _ = self.train_one_epoch(epoch)
            if self.validation_data:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self.validation_loss()
                    val_metrics = training_util.get_metrics(
                        self.model, val_loss, num_batches, reset=True
                    )

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[validation_metric]
                    metric_tracker.add_metric(this_epoch_val_metric)

                    if metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break
            # log the training and validation metrics
            for key, value in training_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            # save the best validation metrics so far
            if metric_tracker.is_best_so_far():
                metrics["best_epoch"] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value
                metric_tracker.best_epoch_metrics = val_metrics
                dump_metrics(
                    os.path.join(
                        self._serialization_dir,
                        f"best-metrics_iteration-{iteration}.json",
                    ),
                    metrics,
                )
            # For each iteration, dump the metrics. Makes it much easier to
            # debug later.
            dump_metrics(
                os.path.join(serialization_dir, f"metrics_epoch_{epoch}.json"), metrics
            )
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)

            self._tensorboard.log_metrics(
                training_metrics,
                val_metrics=val_metrics,
                log_to_console=True,
                epoch=epoch + 1,
            )

            # checkpoint the current model (and update best if is best)
            self._save_checkpoint(epoch, metric_tracker, checkpointer)
            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info(
                "Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time)
            )

            if epoch < self.num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * (
                    (self.num_epochs) / float(epoch + 1) - 1
                )
                formatted_time = str(
                    datetime.timedelta(seconds=int(estimated_time_remaining))
                )
                logger.info("Estimated training time remaining: %s", formatted_time)

        # finally load best model
        best_model_state = checkpointer.best_model_state()
        if best_model_state:
            model_state_dict = best_model_state["model"]
            self.model.load_state_dict(model_state_dict)
        return metrics

    def _save_checkpoint(
        self, epoch: int, metric_tracker: MetricTracker, checkpointer: Checkpointer
    ) -> None:
        training_states = {
            "metric_tracker": metric_tracker.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        model_state = {"model": self.model.state_dict()}
        checkpointer.save_checkpoint(
            model_state=model_state,
            epoch=epoch,
            training_states=training_states,
            is_best_so_far=metric_tracker.is_best_so_far(),
        )

    @overrides
    def train(self) -> Dict[str, Any]:
        for iteration in range(self.num_iterations):
            logger.info(f"Starting Training {iteration + 1} / {self.num_iterations}")
            self.train_one_iteration(iteration)
            # If testing data is available, generate testing metrics
            if self.test_data:
                # Either test_data is a Dict[str, List[Instance]]
                if isinstance(self.test_data, MutableMapping):
                    for key, instance_list in self.test_data.items():
                        test_loss, num_batches = self.validation_loss(
                            instance_list, test=True
                        )
                        test_metrics = training_util.get_metrics(
                            model=self.model,
                            total_loss=test_loss,
                            num_batches=num_batches,
                            reset=True,
                        )
                        metrics_path = os.path.join(
                            self._serialization_dir,
                            f"test-{key}-best_metrics_iteration-{iteration}.json",
                        )
                        dump_metrics(metrics_path, test_metrics)
                else:
                    # Or the test_data is an List[Instance]
                    instance_list: List[Instance] = self.test_data
                    test_loss, num_batches = self.validation_loss(
                        instance_list, test=True
                    )
                    test_metrics = training_util.get_metrics(
                        self.model, test_loss, num_batches, reset=True
                    )
                    metrics_path = os.path.join(
                        self._serialization_dir,
                        f"test-best_metrics_iteration-{iteration}.json",
                    )
                    dump_metrics(metrics_path, test_metrics)

            file_path = os.path.join(
                self._serialization_dir, f"probs_dump_itr-{iteration}.txt"
            )
            (
                self.train_data,
                self.unlabeled_data,
            ) = self.acquisition_function.select_new_data(
                train_data=self.train_data,
                unlabeled_data=self.unlabeled_data,
                file_path=file_path,
            )

    @classmethod
    def from_params(
        cls,
        model: Model,
        serialization_dir: str,
        train_data: Iterable[Instance],
        validation_data: Iterable[Instance],
        unlabeled_data: Iterable[Instance],
        test_data: Union[Iterable[Instance], Dict[str, Iterable]],
        batcher: DataIterator,
        test_batcher: Optional[DataIterator],
        trainer_params: Params,
    ) -> "BaselineActiveLearningTrainer":
        kwargs = {
            "model": model,
            "train_data": train_data,
            "validation_data": validation_data,
            "unlabeled_data": unlabeled_data,
            "test_data": test_data,
            "serialization_dir": serialization_dir,
            "batcher": batcher,
            "test_batcher": test_batcher,
        }
        # move model to GPU
        cuda_device = parse_cuda_device(trainer_params.pop("cuda_device", -1))
        kwargs["cuda_device"] = cuda_device

        kwargs["optimizer_params"] = trainer_params.pop("optimizer", None)

        kwargs["num_iterations"] = trainer_params.pop_int("num_iterations")
        kwargs["num_epochs"] = trainer_params.pop_int("num_epochs")

        kwargs["acquisition_function"] = BaseAcquisitionFunction.from_params(
            trainer_params.pop("acquisition_function"), model=model
        )

        kwargs["patience"] = trainer_params.pop_int("patience", -1)
        kwargs["validation_metric"] = trainer_params.pop("validation_metric", "-loss")

        kwargs["grad_norm"] = trainer_params.pop_float("grad_norm", None)
        kwargs["learning_rate_scheduler_params"] = trainer_params.pop(
            "learning_rate_scheduler_params", None
        )

        trainer_params.assert_empty(cls.__name__)

        return cls(**kwargs)
