local train_data_path = "./data/dependency-parsing/processed-treebanks/UD_English-EWT/en_ewt-ud-train.conllu";
local unlabeled_data_path = "./data/dependency-parsing/processed-treebanks/UD_English-EWT/en_ewt-ud-dev.conllu";
local validation_data_path = "./data/dependency-parsing/processed-treebanks/UD_English-EWT/en_ewt-ud-dev.conllu";
local test_data_path = "./data/dependency-parsing/processed-treebanks/UD_English-EWT/en_ewt-ud-test.conllu";
local bert_dir = "./data/embeddings/mbert/bert-base-multilingual-cased";
local vocab_dir = "./data/dependency-parsing/vocabulary";

{
    "dataset_reader": {
        "type": "albert_dependency_reader",
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": bert_dir + "/vocab.txt",
                "do_lowercase": false
            }
        }
    },

    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "test_data_path": test_data_path,
    "unlabeled_data_path": unlabeled_data_path,

    "model": {
        "type": "custom_dependency_parser",
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": [
                    "bert",
                    "bert-offsets"
                ]
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": bert_dir + "/bert-base-multilingual-cased.tar.gz",
                }
            }
        },
        "encoder": {
            "type": "pass_through",
            "input_dim": 768 + 256
        },
        "tag_representation_dim": 256,
        "arc_representation_dim": 768,
        "pos_embed_dim": 256,
        "dropout": 0.2,
        "decoder": {
            "type": "mst_edges_and_labels",
            "tag_representation_dim": 256
        }
    },

    "vocabulary": {
        "directory_path": vocab_dir,
        "extend": false
    },

    "iterator": {
        "type": "basic",
        "batch_size": 16,
        "maximum_samples_per_batch": ["num_tokens", 16 * 100]
    },

    "trainer_params": {
        "cuda_device": 0,
        "num_iterations": 1,
        "num_epochs": 75,
        "patience": 10,
        "validation_metric": "+LAS",

        "optimizer": {
            "type": "adam",
            "lr": 2e-5
        },
        "acquisition_function": {
            "type": "random",
            "budget": {
                "type": "token",
                "max_instances": 140,
                "max_tokens": 2036
            }
        }
    }
}