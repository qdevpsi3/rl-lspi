<h1 align="center" style="margin-top: 0px;"> <b>Least-Squares Policy Iteration</b></h1>
<div align="center" >

[![paper](https://img.shields.io/static/v1.svg?label=Paper&message=LP03&color=blue)](https://jmlr.csail.mit.edu/papers/v4/lagoudakis03a.html)
[![language](https://img.shields.io/static/v1.svg?label=Language&message=Python&color=blue)](https://www.python.org/)
[![packages](https://img.shields.io/static/v1.svg?label=Made%20with&message=Gym&color=31bac6)](https://gym.openai.com)
[![license](https://img.shields.io/static/v1.svg?label=License&message=GPL%20v3.0&color=blue)](https://www.gnu.org/licenses/gpl-3.0.html)
[![exp_1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qdevpsi3/rl-lspi/blob/main/notebooks/experiment_1.ipynb)
</div>

## Description
This repository contains an implementation of the <ins>model-free</ins> approaches in :

- Paper : **Least-Squares Policy Iteration**
- Authors : **Lagoudakis and Parr**
- Date : **2003**

## Setup
To <ins>install</ins>, clone this repository and execute the following commands :

```
$ cd rl-lspi
$ pip install -r requirements.txt
$ pip install -e .
```

## Details
The available <ins>policy evaluation</ins> methods are :

- **LSTD*Q*** (iterative or by batch)
- **LSTD*Q*-OPT** (iterative, based on Sherman-Morrison formula)

The available <ins>features</ins> are :

- ***Polynomial*** functions
- ***Radial Basis*** functions

The <ins>experiments</ins> in the paper are reproduced for the following environments :

- Experiment 1 : ***Chain Walk*** environment
- Experiment 2 : ***Inverted Pendulum*** environment