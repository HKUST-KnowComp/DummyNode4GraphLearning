vanilla_hyper_params = {
    'GraphSAGE': {
                    'PROTEINS': """python main.py --dataset PROTEINS --model GraphSAGE --seed """,
                    'DD': """python main.py --dataset DD --batch_size 64 --lr 0.0001 --dropout_ratio 0.5 --model GraphSAGE --seed """,
                    'NCI1': """python main.py --dataset NCI1 --model GraphSAGE --seed """,
                    'NCI109': """python main.py --dataset NCI109 --model GraphSAGE --seed """
                },
    'GCN': {
                    'PROTEINS': """python main.py --dataset PROTEINS --model GCN --seed """,
                    'DD': """python main.py --dataset DD --batch_size 64 --lr 0.0001 --dropout_ratio 0.5 --model GCN --seed """,
                    'NCI1': """python main.py --dataset NCI1 --model GCN --seed """,
                    'NCI109': """python main.py --dataset NCI109 --model GCN --seed """
                },
    'GIN': {
                    'PROTEINS': """python main.py --dataset PROTEINS --batch_size 128 --lr 0.01 --weight_decay 0. --hidden_dim 32 --additional '{"train_eps":true,"num_layers":4,"aggregation":"sum"}' --model GIN --seed """,
                    'DD': """python main.py --dataset DD --batch_size 128 --lr 0.01 --dropout_ratio 0.5 --weight_decay 0. --hidden_dim 64 --additional '{"train_eps":true,"num_layers":4,"aggregation":"sum"}' --model GIN --seed """,
                    'NCI1': """python main.py --dataset NCI1 --batch_size 128 --lr 0.01 --weight_decay 0. --model GIN --seed """,
                    'NCI109': """python main.py --dataset NCI109 --batch_size 128 --lr 0.01 --weight_decay 0. --model GIN --seed """
                },
    'RGCN': {
                    'PROTEINS': """python main.py --dataset PROTEINS --model RGCN --seed """,
                    'DD': """python main.py --dataset DD --batch_size 64 --lr 0.0001 --dropout_ratio 0.5 --model RGCN --seed """,
                    'NCI1': """python main.py --dataset NCI1 --batch_size 512 --lr 0.001 --dropout_ratio 0.5 --hidden_dim 32 --model RGCN --seed """,
                    'NCI109': """python main.py --dataset NCI109 --batch_size 512 --lr 0.001 --dropout_ratio 0.0 --hidden_dim 64 --model RGCN --seed """
                },
    'RGIN': {
                    'PROTEINS': """python main.py --dataset PROTEINS --batch_size 128 --lr 0.01 --weight_decay 0. --hidden_dim 32 --model RGIN --seed """,
                    'DD': """python main.py --dataset DD --batch_size 32 --lr 0.01 --dropout_ratio 0.5 --weight_decay 0. --hidden_dim 32 --model RGIN --seed """,
                    'NCI1': """python main.py --dataset NCI1 --batch_size 512 --lr 0.001 --dropout_ratio 0.0 --hidden_dim 64 --additional '{"num_layers":4}' --model RGIN --seed """,
                    'NCI109': """python main.py --dataset NCI109 --batch_size 512 --lr 0.001 --dropout_ratio 0.0 --hidden_dim 64 --additional '{"num_layers":4}' --model RGIN --seed """
                },
    'DiffPool': {
                    'PROTEINS': """python main.py --dataset PROTEINS --model DiffPool --seed """,
                    'DD': """python main.py --dataset DD --batch_size 16 --lr 0.0001 --dropout_ratio 0.5 --model DiffPool --seed """,
                    'NCI1': """python main.py --dataset NCI1 --model DiffPool --seed """,
                    'NCI109': """python main.py --dataset NCI109 --lr 1e-4 --batch_size 128 --model DiffPool --seed """
                },
    'HGP-SL': {
                    'PROTEINS': """python main.py --dataset PROTEINS --model Model --batch_size 128 --seed """,
                    'DD': """python main.py --dataset DD --model Model --batch_size 64 --lr 0.0001 --pooling_ratio 0.3 --dropout_ratio 0.5 --seed """,
                    'NCI1': """python main.py --dataset NCI1 --model Model --pooling_ratio 0.8 --seed """,
                    'NCI109': """python main.py --dataset NCI109 --model Model --pooling_ratio 0.8 --seed """
                }
}



w_dummy_hyper_params = {
    'GraphSAGE': {
                    'PROTEINS': """python main.py --dataset PROTEINS --model GraphSAGE --dummy True --seed """,
                    'DD': """python main.py --dataset DD --batch_size 64 --lr 0.0001 --dropout_ratio 0.5 --model GraphSAGE --dummy True --seed """,
                    'NCI1': """python main.py --dataset NCI1 --model GraphSAGE --dummy True --seed """,
                    'NCI109': """python main.py --dataset NCI109 --model GraphSAGE --dummy True --seed """
                },
    'GCN': {
                    'PROTEINS': """python main.py --dataset PROTEINS --model GCN --dummy True --dummy_weight 0.01 --seed """,
                    'DD': """python main.py --dataset DD --batch_size 64 --lr 0.0001 --dropout_ratio 0.5 --model GCN --dummy True --dummy_weight 10 --seed """,
                    'NCI1': """python main.py --dataset NCI1 --model GCN --dummy True --dummy_weight 0.1 --seed """,
                    'NCI109': """python main.py --dataset NCI109 --model GCN --dummy True --dummy_weight 10 --seed """
                },    
    'GIN': {
                    'PROTEINS': """python main.py --dataset PROTEINS --batch_size 128 --lr 0.01 --weight_decay 0. --hidden_dim 32 --additional '{"train_eps":true,"num_layers":4,"aggregation":"sum"}' --model GIN --dummy True --seed """,
                    'DD': """python main.py --dataset DD --batch_size 128 --lr 0.01 --dropout_ratio 0.5 --weight_decay 0. --hidden_dim 64 --additional '{"train_eps":true,"num_layers":4,"aggregation":"sum"}' --model GIN --dummy True --seed """,
                    'NCI1': """python main.py --dataset NCI1 --model GIN --dummy True --seed """,
                    'NCI109': """python main.py --dataset NCI109 --model GIN --dummy True --seed """
                },
    'RGCN': {
                    'PROTEINS': """python main.py --dataset PROTEINS --model RGCN --dummy True --seed """,
                    'DD': """python main.py --dataset DD --batch_size 64 --lr 0.0001 --dropout_ratio 0.5 --model RGCN --dummy True --seed """,
                    'NCI1': """python main.py --dataset NCI1 --batch_size 512 --lr 0.001 --dropout_ratio 0.5 --hidden_dim 32 --model RGCN --dummy True --seed """,
                    'NCI109': """python main.py --dataset NCI109 --batch_size 512 --lr 0.001 --dropout_ratio 0.0 --hidden_dim 64 --model RGCN --dummy True --seed """
                },
    'RGIN': {
                    'PROTEINS': """python main.py --dataset PROTEINS --batch_size 128 --lr 0.01 --weight_decay 0. --hidden_dim 32 --model RGIN --dummy True --seed """,
                    'DD': """python main.py --dataset DD --batch_size 32 --lr 0.01 --dropout_ratio 0.5 --weight_decay 0. --hidden_dim 32 --model RGIN --dummy True --seed """,
                    'NCI1': """python main.py --dataset NCI1 --batch_size 512 --lr 0.001 --dropout_ratio 0.0 --hidden_dim 64 --additional '{"num_layers":4}' --model RGIN --dummy True --seed """,
                    'NCI109': """python main.py --dataset NCI109 --batch_size 512 --lr 0.001 --dropout_ratio 0.0 --hidden_dim 64 --additional '{"num_layers":4}' --model RGIN --dummy True --seed """
                },
    'DiffPool': {
                    'PROTEINS': """python main.py --dataset PROTEINS --model DiffPool --dummy True --dummy_weight 10 --seed """,
                    'DD': """python main.py --dataset DD --batch_size 16 --lr 0.0001 --dropout_ratio 0.5 --model DiffPool --dummy True --dummy_weight 1 --seed """,
                    'NCI1': """python main.py --dataset NCI1 --model DiffPool --dummy True --dummy_weight 0.1 --seed """,
                    'NCI109': """python main.py --dataset NCI109 --lr 1e-4 --batch_size 128 --model DiffPool --dummy True --seed """
                },
    'HGP-SL': {
                    'PROTEINS': """python main.py --dataset PROTEINS --model Model --batch_size 128 --dummy True --dummy_weight 10 --seed """,
                    'DD': """python main.py --dataset DD --model Model --batch_size 64 --lr 0.0001 --pooling_ratio 0.3 --dropout_ratio 0.5 --dummy True --dummy_weight 10 --seed """,
                    'NCI1': """python main.py --dataset NCI1 --model Model --pooling_ratio 0.8 --dummy True --dummy_weight 1 --seed """,
                    'NCI109': """python main.py --dataset NCI109 --model Model --pooling_ratio 0.8 --dummy True --dummy_weight 0.1 --seed """
                }

}



conj_hyper_params = {
    'RGCN': {
                    'PROTEINS': """python main.py --dataset CONJ_PROTEINS --model RGCN --seed """,
                    'DD': """python main.py --dataset CONJ_DD --batch_size 64 --lr 0.0001 --dropout_ratio 0.5 --model RGCN --seed """,
                    'NCI1': """python main.py --dataset CONJ_NCI1 --batch_size 512 --lr 0.001 --dropout_ratio 0.5 --hidden_dim 32 --model RGCN --seed """,
                    'NCI109': """python main.py --dataset CONJ_NCI109 --batch_size 512 --lr 0.001 --dropout_ratio 0.0 --hidden_dim 64 --model RGCN --seed """
                },
    'RGIN': {
                    'PROTEINS': """python main.py --dataset CONJ_PROTEINS --batch_size 128 --lr 0.01 --weight_decay 0. --hidden_dim 32 --model RGIN --seed """,
                    'DD': """python main.py --dataset CONJ_DD --batch_size 32 --lr 0.01 --dropout_ratio 0.5 --weight_decay 0. --hidden_dim 32 --model RGIN --seed """,
                    'NCI1': """python main.py --dataset CONJ_NCI1 --batch_size 512 --lr 0.001 --dropout_ratio 0.0 --hidden_dim 64 --additional '{"num_layers":4}' --model RGIN --seed """,
                    'NCI109': """python main.py --dataset CONJ_NCI109 --batch_size 512 --lr 0.001 --dropout_ratio 0.0 --hidden_dim 64 --additional '{"num_layers":4}' --model RGIN --seed """
                }
}