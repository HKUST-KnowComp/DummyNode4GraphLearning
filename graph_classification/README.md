## Graph Classification

This part is modified from [sparsewl](https://github.com/chrsmrrs/sparsewl)

### Stage 1: Preprocessing

We conduct experiments on 4 graph benchmark datasets: ```PROTEINS```, ```DD```, ```NCI109```, and ```NCI1```.

Please download data from [TUDatasets](https://chrsmrrs.github.io/datasets/docs/datasets/).

Then unzip data to `data_processing/tu_data` and run `data_processing/tu_data_processing.py` to add with dummy nodes and convert to conjugates.

```bash
cd data_processing
python tu_data_processing.py --load_data_dir tu_data/PROTEINS --save_dummy_data_dir tu_data/DUMMY_PROTEINS --save_line_data_dir tu_data/LINE_PROTEINS --save_conjugate_data_dir tu_data/CONJ_PROTEINS
python tu_data_processing.py --load_data_dir tu_data/DD --save_dummy_data_dir tu_data/DUMMY_DD --save_line_data_dir tu_data/LINE_DD --save_conjugate_data_dir tu_data/CONJ_DD
python tu_data_processing.py --load_data_dir tu_data/NCI1 --save_dummy_data_dir tu_data/DUMMY_NCI1 --save_line_data_dir tu_data/LINE_NCI1 --save_conjugate_data_dir tu_data/CONJ_NCI1
python tu_data_processing.py --load_data_dir tu_data/NCI109 --save_dummy_data_dir tu_data/DUMMY_NCI109 --save_line_data_dir tu_data/LINE_NCI109 --save_conjugate_data_dir tu_data/CONJ_NCI109
```

### Stage 2: Training and Evaluation

#### Kernels
* ```SP```, ```GR```, ```WLOA```, and ```1-WL``` for graph structures
* ```2-WL```, ```δ-2-WL```, ```δ-2-LWL```, and ```δ-2-LWL+``` for tuple-graph structures

Please note that you need to install ```Eigen``` manually.

```bash
cd graph_kernels
g++ gram.cpp src/*cpp -std=c++11 -o gram.out -O2 -I YOUR_CONDA_ENV_PATH/include --static
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
cd graph_kernels
g++ gram.cpp src/*cpp -std=c++11 -o gram.out -O2 -I YOUR_CONDA_ENV_PATH/include --static
python run.py --datasets DUMMY_PROTEINS DUMMY_DD DUMMY_NCI1 DUMMY_NCI109 --dataset_dir ../data_processing/tu_data --kernel SP --k 1 --add_origin true # SP w/ G_varphi
python run.py --datasets DUMMY_PROTEINS DUMMY_DD DUMMY_NCI1 DUMMY_NCI109 --dataset_dir ../data_processing/tu_data --kernel GR --k 1 --add_origin true # GR w/ G_varphi
python run.py --datasets DUMMY_PROTEINS DUMMY_DD DUMMY_NCI1 DUMMY_NCI109 --dataset_dir ../data_processing/tu_data --kernel WLOA --k 1 --add_origin true # WLOA w/ G_varphi
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

#### Models
* ```GraphSAGE```, ```GCN```, and ```GIN``` for homogeneous message-passing implementaions.
* ```RGCN``` and ```RGIN``` for heterogeneous message-passing implementaions.
* ```DiffPool``` and ```HGP-SL``` for pooling-based networks.

