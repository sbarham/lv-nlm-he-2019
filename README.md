# Aggressive Training of Inference Network

This is PyTorch implementation of the [paper](http://arxiv.org/abs/1901.05534):

```
Lagging Inference Networks and Posterior Collapse in Variational Autoencoders
Junxian He, Daniel Spokoyny, Graham Neubig, Taylor Berg-Kirkpatrick
ICLR 2019
```

The code seperates optimization of encoder and decoder in VAE, and performs more steps of encoder update in each iteration. This new training procedure mitigates the issue of posterior collapse in VAE and leads to a better VAE model.

Please contact junxianh@cs.cmu.edu if you have any questions.

## Requirements

* Python 3
* PyTorch 0.4

## Data

Datasets used in this paper can be downloaded with:

```shell
python prepare_data.py
```

Downloaded data is located in `./datasets/`.


## Usage

Example script to train VAE on text data (training uses GPU when available):
```
python text.py --dataset yahoo --aggressive 1 --warm_up 10 --kl_start 0.1
```
image data:
```
python image.py --dataset omniglot --aggressive 1 --warm_up 10 --kl_start 0.1
```

Here:

* `--dataset` specifies the dataset name, currently it supports `synthetic`, `yahoo`, `yelp` for `text.py` and `omniglot` for `image.py`

* `--aggressive` controls whether applies aggressive training or not
* `--kl_start` represents starting KL weight (set to 1.0 to disable KL annealing)
* `--warm_up` represents number of annealing epochs (KL weight increases from `kl_start` to 1.0 linearly in the first `warm_up` epochs)

To run the code on your own text/image dataset, you need to create a new configuration file in `./config/` folder to specifiy network hyperparameters and datapath. If the new config file is `./config/config_abc.py`, then `--dataset` needs to be set as `abc` accordingly.

## Reference

```
@inproceedings{he2018lagging,
title={Lagging Inference Networks and Posterior Collapse in Variational Autoencoders},
author={Junxian He and Daniel Spokoyny and Graham Neubig and Taylor Berg-Kirkpatrick},
booktitle={Proceedings of ICLR},
year={2019}
}
```



