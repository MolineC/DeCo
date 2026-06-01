# DeCo

Official source code for **"Learning to Handle Constrained Routing Problems From a Decoupling Perspective"**.


```
## Quick Start

This repository implements DeCo-based models for **TSPDL** and **TSPTW**.

## Requirements

* Python 3.8+
* Dependencies listed in `requirements.txt`

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training

Example for TSPDL:

```bash
cd DeCo/TSPDL
python train.py
```

Please modify `train.py` as needed to specify the dataset, model configuration, and training settings.

## Testing

Example for TSPDL:

```bash
cd DeCo/TSPDL
python test.py
```

Before running evaluation, set the model checkpoint and data paths in `test.py`, including `model_load_path`, `model_load_epoch`, `data_path`, and `solution_path`.

Example configuration for TSPTW-50:

```python
model_load_path = 'pretrained/Deco/TSPTW/Deco50'
model_load_epoch = 100

data_path = './data/tsptw50_easy.pkl'
solution_path = './data/lkh_tsptw50_easy.pkl'
```
## Citation
if you find this work useful, please cite our paper:
```
@inproceedings{cao2026deco,
  author    = {Rui Cao and Zhiguang Cao and Yihan Huang and Jiaqi Wang and Yuan Jiang and Yubin Xiao and You Zhou},
  title     = {Learning to Handle Constrained Routing Problems From a Decoupling Perspective},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year      = {2026},
}
```

## Acknowledgments

We would like to thank the following repository. Our code is based on their implementation:


https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD

https://github.com/jieyibi/PIP-constraint