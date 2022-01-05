# Deep Multi-View Spatial-Temporal Network for Taxi Demand Prediction

This is a PyTorch implementation of Deep Multi-View Spatial-Temporal Network  in the following paper:

Yao H, Wu F, Ke J, et al.[Deep Multi-View Spatial-Temporal Network for Taxi Demand Prediction](https://arxiv.org/abs/1802.08714)[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2018, 32(1).

## Requirements

* torch >= 1.7
* numpy
* pandas
* h5py
* pickle

**Notice**: When you generate training data, please ensure your server memory is larger than 32G.

### Data Preparation

I use the TaxiBJ Dataset to train the model.

* in every time slot for each grid, generate $S \times S$ data.

### Graph Construction

**Notice：**I use the training data to make graph rather than all data.

* Generate weekly average data for every grid(run `compute_weekly_demand.ipynb`)
* Compute dtw distance in all nodes.(run `lib/compute_dtw.py`)
* Make graph (run `lib/make_graph.py`)
* Use [LINE](https://github.com/tangjianpku/LINE) to get node embedding.

### Experiments

The origin datasets contains $32 \times 32$ grids. In this experiment I use $20 \times 20$ grids, and 1 flow.

|                                     | MAE    | MAPE  | RMSE   |
| :---------------------------------: | ------ | ----- | ------ |
|              DMVSTNet               | 13.461 | 0.351 | 21.854 |
| Temporal view + Spatial (LCNN) view | 14.417 | 0.418 | 23.142 |



### Refered implementations

Tensorflow: https://github.com/huaxiuyao/DMVST-Net

https://github.com/DiamondSick/ST-ResNet-PyTorch