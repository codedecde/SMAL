// Configuration for the NER model with BERT (Delvin et al. 2018)
{
  "base_output_dir": "./trained_models/baselines/bert-tagger",
  "dataset_reader": {
    "type": "conll2003",
    "tag_label": "ner",
    "coding_scheme": "IOB1",
    "token_indexers": {
      "bert": {
          "type": "bert-pretrained",
          "pretrained_model": "./data/embeddings/bert/bert-base-multilingual-cased/vocab.txt",
          "do_lowercase": false,
          "use_starting_offsets": true
      }
    }
  },
  "train_data_path": "./data/conll2003/en/iob1/eng.train",
  "validation_data_path": "./data/conll2003/en/iob1/eng.testa",
  "test_data_path": "./data/conll2003/en/iob1/eng.testb",
  "model": {
    "type": "bert_for_tagging",
    "bert_model": "./data/embeddings/bert/bert-base-multilingual-cased/bert-base-multilingual-cased.tar.gz",
    "top_layer_only": true,
    "calculate_span_f1": true,
    "label_encoding": "IOB1"
  },
  "iterator": {
    "type": "basic",
    "batch_size": 16
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 5e-5
    },
    "validation_metric": "+f1-measure-overall",
    "num_serialized_models_to_keep": 1,
    "num_epochs": 10,
    "cuda_device": 0
  }
}
