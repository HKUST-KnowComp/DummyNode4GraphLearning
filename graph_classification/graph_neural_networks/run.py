import os

run = os.system

folder_dir = './'
os.chdir(folder_dir)

if __name__ == "__main__":

    def seq_tasks(list_of_string):
        for string in list_of_string:
            print('RUNNING:', string)
            run(string)

    # TODO: replace this with contents in hyper_params.py
    default_run_strings = {
        'PROTEINS': 'python main.py',
        'DD': 'python main.py --dataset DD --batch_size 64 --lr 0.0001 --dropout_ratio 0.5',
        'NCI1': 'python main.py --dataset NCI1 ',
        'NCI109': 'python main.py --dataset NCI109 '
    }

    seeds = range(2020, 2030)
    seedstr = ' --seed {seed}'
    modelstr = ' --model {model}'
    savestr = ' --save_model_dir checkpoints/{dataset}/{model}/{seed}'
    ''' GCN exp '''
    # replicated
    model = 'GCN'
    for dataset in default_run_strings:
        runstr = default_run_strings[dataset]
        for n in range(10):
            print(
                'repeat #{}, '.format(n) + runstr +
                (modelstr + seedstr + savestr).format(seed=seeds[n], model=model, dataset=dataset)
            )
            run(runstr + (modelstr + seedstr + savestr).format(seed=seeds[n], model=model, dataset=dataset))

    # varying dummy edge weights
    options = [0.01, 0.1, 1., 10.]
    postfix = ' --add_dummy true --dummy_weight {}'
    for dataset in default_run_strings:
        for option in options:
            runstr = default_run_strings[dataset] + postfix.format(option)
            for n in range(10):
                print(
                    'repeat #{}, '.format(n) + runstr +
                    (modelstr + seedstr + savestr).format(seed=seeds[n], model=model, dataset=dataset)
                )
                run(runstr + (modelstr + seedstr + savestr).format(seed=seeds[n], model=model, dataset=dataset))
    ''' GCN_concat_readout exp '''
    model = 'GCN_concat_readout'
    # replicated
    for dataset in default_run_strings:
        #     if dataset in ['PROTEINS', 'NCI1']: continue
        for postfix in ['']:
            runstr = default_run_strings[dataset] + postfix
            for n in range(10):
                print(
                    'repeat #{}, '.format(n) + runstr +
                    (modelstr + seedstr + savestr).format(seed=seeds[n], model=model, dataset=dataset)
                )
                run(runstr + (modelstr + seedstr + savestr).format(seed=seeds[n], model=model, dataset=dataset))

    # varying dummy edge weights
    options = [0.01, 0.1, 1., 10.]
    postfix = ' --add_dummy true --dummy_weight {}'
    for dataset in default_run_strings:
        for option in options:
            runstr = default_run_strings[dataset] + postfix.format(option)
            for n in range(10):
                print(
                    'repeat #{}, '.format(n) + runstr +
                    (modelstr + seedstr + savestr).format(seed=seeds[n], model=model, dataset=dataset)
                )
                run(runstr + (modelstr + seedstr + savestr).format(seed=seeds[n], model=model, dataset=dataset))
    ''' GraphSAGE exp '''
    model = 'GraphSAGE'
    # replicated
    for dataset in default_run_strings:
        runstr = default_run_strings[dataset] + (modelstr + seedstr + savestr)
        for n in range(10):
            print('repeat #{}, '.format(n) + runstr.format(seed=seeds[n], model=model, dataset=dataset))
            run(runstr.format(seed=seeds[n], model=model, dataset=dataset))

    # dummy
    for dataset in default_run_strings:
        runstr = default_run_strings[dataset] + (modelstr + seedstr + savestr) + ' --add_dummy true'
        for n in range(10):
            print('repeat #{}, '.format(n) + runstr.format(seed=seeds[n], model=model, dataset=dataset))
            run(runstr.format(seed=seeds[n], model=model, dataset=dataset))

    # GraphSAGE + dummy edge weights
    postfix = ' --add_dummy true --dummy_weight {}'
    options = [0.01, 0.1, 1., 10.]
    # replicated
    for dataset in default_run_strings:
        for option in options:
            runstr = default_run_strings[dataset] + (modelstr + seedstr + savestr) + postfix.format(option)
            for n in range(10):
                print('repeat #{}, '.format(n) + runstr.format(seed=seeds[n], model=model, dataset=dataset))
                run(runstr.format(seed=seeds[n], model=model, dataset=dataset))
    ''' GIN exp '''
    model = 'GIN'
    # replicated
    for dataset in default_run_strings:
        runstr = default_run_strings[dataset] + (modelstr + seedstr + savestr)
        for n in range(10):
            print('repeat #{}, '.format(n) + runstr.format(seed=seeds[n], model=model, dataset=dataset))
            run(runstr.format(seed=seeds[n], model=model, dataset=dataset))

    # dummy
    for dataset in default_run_strings:
        runstr = default_run_strings[dataset] + (modelstr + seedstr + savestr) + ' --add_dummy true'
        for n in range(10):
            print('repeat #{}, '.format(n) + runstr.format(seed=seeds[n], model=model, dataset=dataset))
            run(runstr.format(seed=seeds[n], model=model, dataset=dataset))

    postfix = ' --add_dummy true --dummy_weight {}'
    options = [0.01, 0.1, 1., 10.]
    # replicated
    for dataset in default_run_strings:
        for option in options:
            runstr = default_run_strings[dataset] + (modelstr + seedstr + savestr) + postfix.format(option)
            for n in range(10):
                print('repeat #{}, '.format(n) + runstr.format(seed=seeds[n], model=model, dataset=dataset))
                run(runstr.format(seed=seeds[n], model=model, dataset=dataset))
    ''' DiffPool exp '''
    model = 'DiffPool'
    # replicated
    for dataset in default_run_strings:
        runstr = default_run_strings[dataset] + (modelstr + seedstr + savestr)
        for n in range(10):
            print('repeat #{}, '.format(n) + runstr.format(seed=seeds[n], model=model, dataset=dataset))
            run(runstr.format(seed=seeds[n], model=model, dataset=dataset))
