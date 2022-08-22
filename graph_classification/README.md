## Graph Classification

This part is modified from [sparsewl](https://github.com/chrsmrrs/sparsewl) and [HGP-SL](https://github.com/cszhangzhen/HGP-SL)

### Stage 1: Preprocessing

We conduct experiments on 4 graph benchmark datasets: ```PROTEINS```, ```DD```, ```NCI109```, and ```NCI1```.

Run the following scrips to download data, add with dummy nodes, and convert to conjugates.

```bash
python data_processing/tu_data_processing.py --dataset PROTEINS --data_dir data_processing/tu_data
python data_processing/tu_data_processing.py --dataset DD --data_dir data_processing/tu_data
python data_processing/tu_data_processing.py --dataset NCI1 --data_dir data_processing/tu_data
python data_processing/tu_data_processing.py --dataset NCI109 --data_dir data_processing/tu_data
```

### Stage 2: Training and Evaluation

#### Kernels
* ```SP```, ```GR```, ```WLOA```, and ```1-WL``` for graph structures
* ```2-WL```, ```δ-2-WL```, ```δ-2-LWL```, and ```δ-2-LWL+``` for tuple-graph structures

Please note that you need to install ```Eigen``` manually.

```bash
cd graph_kernels
g++ gram.cpp src/*cpp -std=c++11 -o gram.out -O2 -I YOUR_INCLUDE_PATH_WITH_EIGEN --static
```

```bash
python run.py --datasets PROTEINS DD NCI1 NCI109 --dataset_dir ../data_processing/tu_data --kernel SP --k 1 --add_origin false # SP w/ G
python run.py --datasets PROTEINS DD NCI1 NCI109 --dataset_dir ../data_processing/tu_data --kernel GR --k 1 --add_origin false # GR w/ G
python run.py --datasets PROTEINS DD NCI1 NCI109 --dataset_dir ../data_processing/tu_data --kernel WLOA --k 1 --add_origin false # WLOA w/ G
python run.py --datasets PROTEINS DD NCI1 NCI109 --dataset_dir ../data_processing/tu_data --kernel WL --k 1 --add_origin false # 1-WL w/ G
python run.py --datasets PROTEINS DD NCI1 NCI109 --dataset_dir ../data_processing/tu_data --kernel WL --k 2 --add_origin false # 2-WL w/ G
python run.py --datasets PROTEINS DD NCI1 NCI109 --dataset_dir ../data_processing/tu_data --kernel DWL --k 2 --add_origin false # δ-2-WL w/ G
python run.py --datasets PROTEINS DD NCI1 NCI109 --dataset_dir ../data_processing/tu_data --kernel LWL --k 2 --add_origin false # δ-2-LWL w/ G
python run.py --datasets PROTEINS DD NCI1 NCI109 --dataset_dir ../data_processing/tu_data --kernel LWLP --k 2 --add_origin false # δ-2-LWL+ w/ G
```

To conduct the experiments with dummy nodes, please use the datasets with dummy and set `--add_origin True`

```bash
python run.py --datasets DUMMY_PROTEINS DUMMY_DD DUMMY_NCI1 DUMMY_NCI109 --dataset_dir ../data_processing/tu_data --kernel SP --k 1 --add_origin true # SP w/ G_varphi
python run.py --datasets DUMMY_PROTEINS DUMMY_DD DUMMY_NCI1 DUMMY_NCI109 --dataset_dir ../data_processing/tu_data --kernel GR --k 1 --add_origin true # GR w/ G_varphi
python run.py --datasets DUMMY_PROTEINS DUMMY_DD DUMMY_NCI1 DUMMY_NCI109 --dataset_dir ../data_processing/tu_data --kernel WL --k 1 --add_origin true # 1-WL w/ G_varphi
python run.py --datasets DUMMY_PROTEINS DUMMY_DD DUMMY_NCI1 DUMMY_NCI109 --dataset_dir ../data_processing/tu_data --kernel WL --k 2 --add_origin true # 2-WL w/ G_varphi
python run.py --datasets DUMMY_PROTEINS DUMMY_DD DUMMY_NCI1 DUMMY_NCI109 --dataset_dir ../data_processing/tu_data --kernel DWL --k 2 --add_origin true # δ-2-WL w/ G_varphi
python run.py --datasets DUMMY_PROTEINS DUMMY_DD DUMMY_NCI1 DUMMY_NCI109 --dataset_dir ../data_processing/tu_data --kernel LWL --k 2 --add_origin true # δ-2-LWL w/ G_varphi
python run.py --datasets DUMMY_PROTEINS DUMMY_DD DUMMY_NCI1 DUMMY_NCI109 --dataset_dir ../data_processing/tu_data --kernel LWLP --k 2 --add_origin true # δ-2-LWL+ w/ G_varphi
```

To conduct the experiments with conjugate graphs, please use the corresponding datasets and set `--add_origin True`

```bash
cd graph_kernels
g++ gram.cpp src/*cpp -std=c++11 -o gram.out -O2 -I YOUR_CONDA_ENV_PATH/include --static
python run.py --datasets CONJ_PROTEINS CONJ_DD CONJ_NCI1 CONJ_NCI109 --dataset_dir ../data_processing/tu_data --kernel SP --k 1 --add_origin true # SP w/ H_Phi
python run.py --datasets CONJ_PROTEINS CONJ_DD CONJ_NCI1 CONJ_NCI109 --dataset_dir ../data_processing/tu_data --kernel GR --k 1 --add_origin true # GR w/ H_Phi
python run.py --datasets CONJ_PROTEINS CONJ_DD CONJ_NCI1 CONJ_NCI109 --dataset_dir ../data_processing/tu_data --kernel WLOA --k 1 --add_origin true # WLOA w/ H_Phi
python run.py --datasets CONJ_PROTEINS CONJ_DD CONJ_NCI1 CONJ_NCI109 --dataset_dir ../data_processing/tu_data --kernel WL --k 1 --add_origin true # 1-WL w/ H_Phi
python run.py --datasets CONJ_PROTEINS CONJ_DD CONJ_NCI1 CONJ_NCI109 --dataset_dir ../data_processing/tu_data --kernel WL --k 2 --add_origin true # 2-WL w/ H_Phi
python run.py --datasets CONJ_PROTEINS CONJ_DD CONJ_NCI1 CONJ_NCI109 --dataset_dir ../data_processing/tu_data --kernel DWL --k 2 --add_origin true # δ-2-WL w/ H_Phi
python run.py --datasets CONJ_PROTEINS CONJ_DD CONJ_NCI1 CONJ_NCI109 --dataset_dir ../data_processing/tu_data --kernel LWL --k 2 --add_origin true # δ-2-LWL w/ H_Phi
python run.py --datasets CONJ_PROTEINS CONJ_DD CONJ_NCI1 CONJ_NCI109 --dataset_dir ../data_processing/tu_data --kernel LWLP --k 2 --add_origin true # δ-2-LWL+ w/ H_Phi
```

#### Graph Neural Networks
* ```GraphSAGE```, ```GCN```, and ```GIN``` for homogeneous message-passing implementaions.
* ```RGCN``` and ```RGIN``` for heterogeneous message-passing implementaions.
* ```DiffPool``` and ```HGP-SL``` for pooling-based networks.

Please note that you need to install ```PYG``` manually.

```bash
cd graph_neural_networks
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model GraphSAGE # GraphSAGE
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model GCN # GCN
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model GIN # GIN
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model RGCN # RGCN
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model RGIN # RGIN
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model DiffPool # DiffPool
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model HGPSLPool # HGP-SL
```

To conduct the experiments with dummy nodes, please use the datasets with dummy and set `--add_dummy true`

```bash
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model GraphSAGE --add_dummy true # GraphSAGE
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model GCN --add_dummy true # GCN
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model GIN --add_dummy true # GIN
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model RGCN --add_dummy true # RGCN
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model RGIN --add_dummy true # RGIN
python main.py --dataset RGCNPROTEINS --dataset_dir ../data_processing/tu_data --model DiffPool --add_dummy true # DiffPool
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model HGPSLPool --add_dummy true # HGP-SL
```

To conduct the experiments with conjugate graphs, please use the corresponding datasets and set `--add_dummy true --convert_conjugate true`

```bash
cd graph_neural_networks
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model GraphSAGE --add_dummy true --convert_conjugate true # GraphSAGE
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model GCN --add_dummy true --convert_conjugate true # GCN
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model GIN --add_dummy true --convert_conjugate true # GIN
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model RGCN --add_dummy true --convert_conjugate true # RGCN
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model RGIN --add_dummy true --convert_conjugate true # RGIN
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model DiffPool --add_dummy true --convert_conjugate true # DiffPool
python main.py --dataset PROTEINS --dataset_dir ../data_processing/tu_data --model HGPSLPool --add_dummy true --convert_conjugate true # HGP-SL
```

More hyper-parameter details can be found in `graph_neural_networks/main.py` and `graph_neural_networks/hyper_params.py`.
