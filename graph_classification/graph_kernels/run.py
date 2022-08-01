import argparse
import os
import sys
import datetime
import subprocess


class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        self.terminal.write(message)
        self.log.write(message)

    def flush(self, *args, **kw):
        self.terminal.flush()
        self.log.flush()


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gram_dir", type=str,
        default=os.path.join(dirname, "gram"),
        help="the directory of gram matrices"
    )
    parser.add_argument(
        "--dataset_dir", type=str,
        default="../data_processing/tu_data",
        help="the directory of gram matrices"
    )
    parser.add_argument(
        "--log_file", type=str,
        default=os.path.join(dirname, "log.txt"),
        help="the log file"
    )
    parser.add_argument(
        "--k", type=int,
        default=1,
        help="complexity of kernel functions"
    )
    parser.add_argument(
        "--n_iters", type=int,
        default=5,
        help="number of runs"
    )
    parser.add_argument(
        "--n_folds", type=int,
        default=10,
        help="folds of cross-validation"
    )
    parser.add_argument(
        "--seeds", nargs="+",
        default=list(map(str, range(2020, 2030))),
        help="seeds"
    )
    parser.add_argument(
        "--kernel", type=str,
        default="WL",
        help="kernel function"
    )
    parser.add_argument(
        "--add_origin", type=str,
        default="false",
        help="whether to add origional gram matrices (true/false)"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        help="datasets seperated by spaces (e.g., ENZYMES IMDB-BINARY MUTAG)"
    )
    args = parser.parse_args()
    args.gram_dir = os.path.abspath(args.gram_dir)
    args.dataset_dir = os.path.abspath(args.dataset_dir)
    args.log_file = os.path.abspath(args.log_file)

    if not os.path.exists(args.gram_dir):
        os.makedirs(args.gram_dir)

    ts = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    sys.stdout = Logger(args.log_file)
    print("-" * 80, flush=True)
    print(ts, flush=True)
    print(" ".join(sys.argv), flush=True)

    # run gram
    processes = []
    for dataset in args.datasets:
        proc = subprocess.Popen(
            [
                "./gram.out",
                "--dataset_dir", args.dataset_dir,
                "--gram_dir", args.gram_dir,
                "--k", str(args.k),
                "--n_iters", str(args.n_iters),
                "--kernel", args.kernel,
                "--datasets", dataset
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        for line in proc.stdout:
            print(line.decode("utf-8"), end="", flush=True)
        processes.append(proc)

    for proc in processes:
        proc.wait()
    processes.clear()

    # add original gram matrices
    if args.add_origin == "true":
        kernel = args.kernel
        if kernel == "WL" or args.k != 1:
            kernel = kernel + str(args.k)
        processes = []
        for dataset in args.datasets:
            for i in range(0, args.n_iters+1):
                new_gram_file = os.path.join(args.gram_dir, dataset + "__" + kernel + "_" + str(i) + ".gram")
                if not os.path.exists(new_gram_file):
                    continue
                if dataset.startswith("LINE_"):
                    origional_dataset = dataset[5:]
                elif dataset.startswith("DUMMY_"):
                    origional_dataset = dataset[6:]
                elif dataset.startswith("CONJ_"):
                    origional_dataset = dataset[5:]
                else:
                    raise ValueError
                origional_gram_file = os.path.join(args.gram_dir, origional_dataset + "__" + kernel + "_" + str(i) + ".gram")

                proc = subprocess.Popen(
                    [
                        "python", "merge_grams.py",
                        "--load_feat_files", origional_gram_file, new_gram_file,
                        "--save_feat_file", new_gram_file
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )
                for line in proc.stdout:
                    print(line.decode("utf-8"), end="", flush=True)
                processes.append(proc)

        for proc in processes:
            proc.wait()
        processes.clear()

    # run svm
    processes = []
    # os.chdir("svm")
    for dataset in args.datasets:
        if args.kernel in ["SP", "GR"]:
            k = 0
        else:
            k = args.k
        proc = subprocess.Popen(
            [
                "python", "seed_svm.py",
                "--dataset_dir", args.dataset_dir,
                "--gram_dir", args.gram_dir,
                "--k", k,
                "--n_iters", str(args.n_iters),
                "--kernel", args.kernel,
                "--datasets", dataset,
                "--seeds"
            ] + args.seeds,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        for line in proc.stdout:
            print(line.decode("utf-8"), end="", flush=True)
        processes.append(proc)

    for proc in processes:
        proc.wait()
    processes.clear()

    print("=" * 80, flush=True)
