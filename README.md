# End-to-End Learning of Behavioural Inputs for Autonomous Driving in Dense Traffic

This repository contains the source code to reproduce the experiments in our [IROS 2023 submission](https://youtu.be/Vr9p_rWRPuM).

![Repo Overview](https://user-images.githubusercontent.com/38403732/223746011-2228a674-08fc-43cf-999a-5abf9c044135.png)

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

![IROS Benchmark](https://user-images.githubusercontent.com/38403732/223748323-672a7999-c74e-4192-8401-075ad0b9b94e.png)

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

1. Download the [training dataset](https://owncloud.ut.ee/owncloud/s/YgdSoGHgX7maSPc) and extract the files to the dataset directory. 

2. The training example is shown in the [Jupyter Notebook](https://github.com/jatan12/DiffProj/blob/main/Behavioral%20Input%20Distribution%20Training.ipynb) and can also be viewed using [Notebook Viewer](https://nbviewer.org/github/jatan12/DiffProj/blob/main/Behavioral%20Input%20Distribution%20Training.ipynb).
