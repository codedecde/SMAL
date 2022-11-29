Use this sub-module to process your data. For example, from the root folder, for dependency parsing:

```
python3.6 -m  albert.data_process.convert_dependency_parsing_datasets --input_dir ./data/dependency-parsing/treebanks/ --output_dir ./data/dependency-parsing/processed_treebanks/
python3.6 -m  albert.data_process.generate_dependency_parsing_subsets --parent_path ./data/dependency-parsing/ --token_frac $tok_frac
```
