import os
run = os.system

folder_dir = '/path/to/graph_neural_networks'
os.chdir(folder_dir)

def seq_tasks(list_of_string):
    for string in list_of_string:
        print('RUNNING:', string)
        run(string)
        
default_run_strings = {
    'PROTEINS': 'python3 main.py',
    'DD': 'python3 main.py --dataset DD --batch_size 64 --lr 0.0001 --dropout_ratio 0.5',
    'NCI1': 'python3 main.py --dataset NCI1 ',
    'NCI109': 'python3 main.py --dataset NCI109 '
}

seeds = range(2020, 2030)

''' baseline_GCN exp '''
seedstr = ' --seed {}'
modelstr = ' --model GCN'
# replicated
for dataset in default_run_strings:
    runstr = default_run_strings[dataset] + modelstr 
    for n in range(10):
        print('repeat #{}, '.format(n)+ runstr+seedstr.format(seeds[n]))
        run(runstr+seedstr.format(seeds[n]))
            
# varying dummy edge weights
options = [0.01, 0.1, 1., 10.]
postfix = ' --dummy True --dummy_weight {}'
for dataset in default_run_strings:
    for option in options:
        runstr = default_run_strings[dataset] + modelstr + postfix.format(option)
        for n in range(10):
            print('repeat #{}, '.format(n)+ runstr+seedstr.format(seeds[n]))
            run(runstr+seedstr.format(seeds[n]))

''' baseline_GCN_concat_readout exp '''  
seedstr = ' --seed {}'
modelstr = ' --model GCN_concat_readout'
# replicated
for dataset in default_run_strings:
#     if dataset in ['PROTEINS', 'NCI1']: continue
    for postfix in ['']:
        runstr = default_run_strings[dataset] + modelstr + postfix
        for n in range(10):
            print('repeat #{}, '.format(n)+ runstr+seedstr.format(seeds[n]))
            run(runstr+seedstr.format(seeds[n]))
            
# varying dummy edge weights
options = [0.01, 0.1, 1., 10.]
postfix = ' --dummy True --dummy_weight {}'
for dataset in default_run_strings:
    for option in options:
        runstr = default_run_strings[dataset] + modelstr + postfix.format(option)
        for n in range(10):
            print('repeat #{}, '.format(n)+ runstr+seedstr.format(seeds[n]))
            run(runstr+seedstr.format(seeds[n]))      
            
''' baseline_GraphSAGE exp '''
seedstr = ' --seed {}'
modelstr = ' --model GraphSAGE'
# replicated
for dataset in default_run_strings:
    runstr = default_run_strings[dataset] + modelstr +seedstr
    for n in range(10):
        print('repeat #{}, '.format(n)+runstr.format(seeds[n]))
        run(runstr.format(seeds[n]))
        
# dummy
for dataset in default_run_strings:
    runstr = default_run_strings[dataset] + modelstr +seedstr + ' --dummy True'
    for n in range(10):
        print('repeat #{}, '.format(n)+runstr.format(seeds[n]))
        run(runstr.format(seeds[n]))
        
# GraphSAGE + dummy edge weights
seedstr = ' --seed {}'
modelstr = ' --model baseline_GraphSAGE'
postfix = ' --dummy True --dummy_weight {}'
options = [0.01, 0.1, 1., 10.]
# replicated
for dataset in default_run_strings:
    for option in options:
        runstr = default_run_strings[dataset] + modelstr +seedstr + postfix.format(option)
        for n in range(10):
            print('repeat #{}, '.format(n)+runstr.format(seeds[n]))
            run(runstr.format(seeds[n]))
        
''' baseline_GIN exp '''
seedstr = ' --seed {}'
modelstr = ' --model GIN'
# replicated
for dataset in default_run_strings:
    runstr = default_run_strings[dataset] + modelstr +seedstr
    for n in range(10):
        print('repeat #{}, '.format(n)+runstr.format(seeds[n]))
        run(runstr.format(seeds[n]))

# dummy
for dataset in default_run_strings:
    runstr = default_run_strings[dataset] + modelstr +seedstr + ' --dummy True'
    for n in range(10):
        print('repeat #{}, '.format(n)+runstr.format(seeds[n]))
        run(runstr.format(seeds[n]))

seedstr = ' --seed {}'
modelstr = ' --model baseline_GIN'
postfix = ' --dummy True --dummy_weight {}'
options = [0.01, 0.1, 1., 10.]
# replicated
for dataset in default_run_strings:
    for option in options:
        runstr = default_run_strings[dataset] + modelstr +seedstr + postfix.format(option)
        for n in range(10):
            print('repeat #{}, '.format(n)+runstr.format(seeds[n]))
            run(runstr.format(seeds[n]))
        
''' baseline_DiffPool exp '''
seedstr = ' --seed {}'
modelstr = ' --model DiffPool'
# replicated
for dataset in default_run_strings:
    runstr = default_run_strings[dataset] + modelstr +seedstr
    for n in range(10):
        print('repeat #{}, '.format(n)+runstr.format(seeds[n]))
        run(runstr.format(seeds[n]))