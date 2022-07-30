## Graph Classification on Graph Neural Networks

### Stage 1: Preparing data

We conduct experiments on 4 graph benchmark datasets: ```PROTEINS```, ```DD```, ```NCI109```, and ```NCI1```.

The processed datasets are already provided in `data.zip`. You may unzip it and put the `data` folder under `graph_neural_networks`.
```bash
unzip data.zip
```

### Stage 2: Training and Evaluation
To run training and evaluation for the vanilla `GCN` model on the ```PROTEINS``` dataset, try the following command

```bash
python3 main.py --model GCN --dataset PROTEINS --seed 2022
```

Further, to test the corresponding performance with dummy nodes, consider the option ``--dummy True``:
```bash
python3 main.py --model GCN --dataset PROTEINS --dummy True --seed 2022
```

For more available options, see
```bash
python3 main.py --help
```
Example running commands are provided in `run.py`.
