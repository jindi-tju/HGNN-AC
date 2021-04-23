# Heterogeneous Graph Neural Network via Attribute Completion
This repository contains the demo code of the paper:
>[Heterogeneous Graph Neural Network via Attribute Completion](https://camps.aptaracorp.com/ACM_PMS/PMS/ACM/WWW21/129/450f1572-6b1a-11eb-8d84-166a08e17233/OUT/www21-129.pdf)

which has been accepted by *WWW2021*.
## Dependencies
* Python3
* NumPy
* SciPy
* scikit-learn
* NetworkX
* DGL
* PyTorch
## Datasets
The preprocessed datasets are available at [Baidu Netdisk](https://pan.baidu.com/s/1teLcrdVxrE1YQVU14sRJyw)(password: hgnn) or [Google Drive](https://drive.google.com/file/d/1PqUjvSViICa8yOszqDrw-j96hXVJ0MHR/view?usp=sharing).

Please extract the zip file to folder `data`.

## Example
* `python run_DBLP.py`
* `python run_IMDB.py`
* `python run_ACM.py`

Please refer to the code for detailed parameters.
## Acknowledgements
The demo code is implemented based on [MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding](https://github.com/cynricfu/MAGNN).
## Citing
    @inproceedings{hgnn-ac,
     title={Heterogeneous Graph Neural Network via Attribute Completion},
     author={Di Jin and Cuiying Huo and Chundong Liang and Liang Yang},
     booktitle = {WWW},
     year={2021}
    }
