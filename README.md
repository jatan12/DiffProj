# End-to-End Learning of Behavioural Inputs for Autonomous Driving in Dense Traffic

This repository contains the source code to reproduce the experiments in our [IROS 2023 paper](https://arxiv.org/abs/2310.14766). [Video of results](https://youtu.be/Vr9p_rWRPuM)

![IROS2023 Overview_page-0001](https://github.com/jatan12/MPC-Bi-Level/assets/38403732/b3ba073c-8064-4819-9baa-543830ac813b)

## Getting Started

1. Clone this repository:

```
git clone https://github.com/jatan12/DiffProj.git
cd DiffProj
```
2. Create a conda environment and install the dependencies:

```
conda create -n diffproj python=3.8
conda activate diffproj
pip install -r requirements.txt
```
3. Download [Trained Models](https://owncloud.ut.ee/owncloud/s/YgdSoGHgX7maSPc) to the weights directory. 

## Reproducing our main experimental results

![IROS Benchmark_page-0001](https://github.com/jatan12/MPC-Bi-Level/assets/38403732/8151ecff-bb62-4692-80a6-855b019df67d)

### Ours

Four Lane
```
python main_diffproj.py --density ${select} --render True
```

Two Lane
```
python main_diffproj.py --density ${select} --two_lane True --render True
```

### Baselines

To run a baseline {batch, grid, mppi}:

Four Lane
```
python main_baseline.py --baseline ${select} --density ${select} --render True
```

Two Lane
```
python main_baseline.py --baseline ${select} --density ${select} --two_lane True --render True
```

## Training the Behavioral Input Distribution Model

![IROS2023 Pipeline_page-0001](https://github.com/jatan12/MPC-Bi-Level/assets/38403732/adf32e92-c89e-4b34-ac49-ed3a9241babd)

1. Download the [training dataset](https://owncloud.ut.ee/owncloud/s/YgdSoGHgX7maSPc) and extract the files to the dataset directory. 

2. The training example is shown in the [Jupyter Notebook](https://github.com/jatan12/DiffProj/blob/main/Behavioral%20Input%20Distribution%20Training.ipynb) and can also be viewed using [Notebook Viewer](https://nbviewer.org/github/jatan12/DiffProj/blob/main/Behavioral%20Input%20Distribution%20Training.ipynb).
