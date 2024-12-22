# TAD-Graph
TAD-Graph: Enhancing Whole Slide Image Analysis via Task-Aware Subgraph Disentanglement


### Installation
```
conda create -n wsi python=3.9
conda activate wsi
sh INSTALL_TORCH.sh
pip install -e .
```

### Dataset Preparation

We follow [CLAM](https://github.com/mahmoodlab/CLAM) to divide each WSI into patches and extract features for each slide. 

1. For patching: 

```
cd src/tadgraph/preprocess
python create_patches_fp.py --task tcga_prad --seg --patch --stitch 
```

2. Extract patch embeddings:

```
cd src/tadgraph/embedder
CUDA_VISIBLE_DEVICES=6,7 python extract_features_fp.py --task cptac_brca --data_h5_dir extracted_mag20x_patch256 --model uni --batch_size 256
```


### Usage
```
cd scripts
```
Please refers to the command in `main.py`.

For example: 
```
CUDA_VISIBLE_DEVICES=6 python main.py --model_name tadgraph --config tad_graph_config.yaml \
--dataset tcga_brca --task her2 --split_file tcga_brca_her2_5fold_val0.2_test0.2_100_seed1  \
--feat_dir extracted_mag20x_patch256/vits_tcga_pancancer_dino_pt_patch_features/slide_graph --embed_size 384 --use_graph \
--lambda_sup 1 --lambda_info 0.5 --lambda_unif 0.5
```


### Acknowledgement

We thank the awesome [CLAM](https://github.com/mahmoodlab/CLAM) and [HIPT](https://github.com/mahmoodlab/HIPT) codebase. 