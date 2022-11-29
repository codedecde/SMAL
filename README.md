# SMAL
Single Model Active Learning

## Instructions

1. Setup a new conda environment with Python 3.6 and pytorch 1.3.1: `conda create -y -q -n smal_env python=3.6`
2. Run `conda install pytorch==1.5.0 torchvision cudatoolkit=10.1 -c pytorch`
3. Install the requirements: `pip: -r requirements.txt`
4. Process your datasets (Eg: refer the `data/` directory for NER and the `albert.data_process` sub-module for classification and dependency parsing)
5. Run `bash get_embeddings.sh --type=mbert`. Note that for some embeddings like fasttext, you might have to also specify the language as `--lang=en` (with the appropriate language(s) to download the embeddings)
6. Run `python3.6 -m albert.baseline_active_learning_main --config_file path/to/config/`

## References

If you found the resources in this paper or repository useful, please cite [On Efficiently Acquiring Annotations for Multilingual Models](https://aclanthology.org/2022.acl-short.9/):

```
@inproceedings{moniz2022efficiently,
  title={On Efficiently Acquiring Annotations for Multilingual Models},
  author={Moniz, Joel and Patra, Barun and Gormley, Matthew R},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  pages={69--85},
  year={2022}
}
```
