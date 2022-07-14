import argparse
import os
import sklearn.datasets as ds
import numpy as np


def read_lib_svm(file_name):
    gram_matrix, labels = ds.load_svmlight_file(file_name, zero_based=True, multilabel=False)
    return gram_matrix.toarray(), labels


def write_lib_svm(file_name, gram_matrix, labels):
    ds.dump_svmlight_file(gram_matrix, labels, file_name, zero_based=True, multilabel=False)


def merge_svm_files(input_file_names, output_file_name):
    assert len(input_file_names) > 0
    gram_matrix, labels = read_lib_svm(input_file_names[0])
    for file_name in input_file_names[1:]:
        gram_matrix_, labels_ = read_lib_svm(file_name)
        assert np.equal(labels, labels_).all()
        gram_matrix += gram_matrix_
    # gram_matrix /= len(input_file_names)
    write_lib_svm(output_file_name, gram_matrix, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_feat_files", nargs="+")
    parser.add_argument("--save_feat_file", type=str)
    args = parser.parse_args()

    merge_svm_files(args.load_feat_files, args.save_feat_file)
