local train_data_path = "";
local unlabeled_data_path = "";
local validation_data_path = "";
local test_data_path = "";
local bert_dir = "./data/embeddings/bert/bert-base-multilingual-cased";

{
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "test_data_path": test_data_path,
    "unlabeled_data_path": unlabeled_data_path,

    "dataset_reader": {
        "type": "albert_classification_reader",
        "bert_model": bert_dir
    },

    "model": {
        "type": "bert_classifier",
        "bert_model": bert_dir + "/bert-base-multilingual-cased.tar.gz",
        "num_labels": 2
    },
    "iterator": {
        "type": "random",
        "batch_size": 32
    },
    "trainer_params": {
        "cuda_device": 0,
        "num_iterations": 4,
        "num_epochs": 10,
        "patience": 2,
        "validation_metric": "+accuracy",
        "optimizer": {
            "type": "adam",
            "lr": 2e-7
        },
        "acquisition_function": {
            "type": "classification_least_confidence",
            "budget": {
                "type": "token",
                "max_instances": 140,
                "max_tokens": 2036
            }
        }
    }
}