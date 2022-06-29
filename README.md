# Feature-Indistinguishable Attack

## Introduction

This repository contains the code that implements the *Feature-Indistinguishable Attack* (*FIA*) described in our paper ["Feature-Indistinguishable Attack to Circumvent Trapdoor-Enabled Defense"](https://www.researchgate.net/publication/356203154_Feature-Indistinguishable_Attack_to_Circumvent_Trapdoor-Enabled_Defense) published at ACM CCS 2021. FIA aims to craft adversarial examples indistinguishable in the feature (i.e., neuron-activation) space from benign examples in the target category. It can successfully circumvent the [Trapdoor-enabled Defense](https://github.com/Shawn-Shan/trapdoor) proposed in the paper ``Gotta Catch’Em All: Using Honeypots to Catch Adversarial Attacks on Neural Networks" published at ACM CCS 2020. FIA opens a door for developing much more powerful adversarial attacks.

## Requirements

Our code is implemented and tested on Keras and TensorFlow with the following packages and versions:

- `python=3.8`
- `tensorflow=1.15.4`
- `keras=2.3.1`
- `cleverhans=3.0.1`

## Quick Start

You can set all configurations (including datasets & our attacks & the trapdoor-enable defense & training settings) with the yaml file in directory 'configs/dataset', where the configurations of the trapdoor-enable defense to protect a single category with a single trapdoor for MNIST and Cifar10 are given by default.

To run the code:

`$ python run_batch.py --config configs/mnist.yaml --device 0`  
(suppose GPU 0 is used).

## Citation

If you are using our code for research purpose, please cite our paper.

```
@inproceedings{fia_ccs2021,
  author    = {Chaoxiang He, Bin Zhu, Xiaojing Ma, Hai Jin, and Shengshan Hu},
  title     = {Feature-Indistinguishable Attack to Circumvent Trapdoor-enabled Defense},
  booktitle = {{CCS} '21: 2021 {ACM} {SIGSAC} Conference on Computer and Communications
               Security, Virtual Event, Republic of Korea, November 15–19, 2021},
  publisher = {{ACM}},
  year      = {2021},
  url       = {https://doi.org/10.1145/3460120.3485378}
}
```
