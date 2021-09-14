# FeatureIndistinguishableAttack
Implementation of our ACM CCS 2021 paper "Feature Indistinguishable Attack to Circumvent Trapdoor-enabled Defense".

## Requirements
- python=3.8.5
- tensorflow=1.15.4
- keras=2.3.1
- cleverhans=3.0.1

## Quick Start
You can set all configurations (including dataset & attacks & trapdoor-enable defense & training settings) with yaml file in directory 'configs/dataset', where single trapdoor single category configurations of mnist and cifar10 are given by default.

To run the code:

`$ python run_batch.py --config configs/mnist.yaml --device 0`

## ACM Reference Format
```
Chaoxiang He, Bin Zhu, Xiaojing Ma, Hai Jin, and Shengshan Hu. 2021. Feature Indistinguishable Attack to Circumvent Trapdoor-enabled Defense. In Proceedings of the 2021 ACM SIGSAC Conference on Computer and Commu- nications Security (CCS ’21), November 15–19, 2021, Virtual Event, Republic of Korea. ACM, New York, NY, USA, 18 pages. https://doi.org/10.1145/3460120. 3485378'
```
