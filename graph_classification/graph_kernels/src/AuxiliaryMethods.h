#ifndef WLFAST_AUXILIARYMETHODS_H
#define WLFAST_AUXILIARYMETHODS_H

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "Graph.h"

using Eigen::IOFormat;
using Eigen::MatrixXd;
using namespace std;
using namespace GraphLibrary;

#ifdef __linux__
#include <eigen3/unsupported/Eigen/src/SparseExtra/MarketIO.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#else
#include "/usr/local/include/eigen3/Eigen/Dense"
#include "/usr/local/include/eigen3/Eigen/Sparse"
#include "/usr/local/include/eigen3/unsupported/Eigen/src/SparseExtra/MarketIO.h"
#endif

namespace AuxiliaryMethods
{
// Simple function for converting a comma separated string into a vector of
// integers.
vector<int> split_string(string s);

// Simple function for converting a comma separated string into a vector of
// floats.
vector<float> split_string_float(string s);

// Reading a graph database from txt file.
GraphDatabase read_graph_txt_file(string path, string data_set_name);

vector<int> read_classes(string path, string data_set_name);

vector<vector<float>> read_multi_targets(string path, string data_set_name);

// Write Gram matrix to file.
void write_gram_matrix(const GramMatrix &gram_matrix, string file_name);

void write_sparse_gram_matrix(const GramMatrix &gram_matrix, string file_name);

void write_libsvm(const GramMatrix &gram_matrix, const vector<int> classes, string filename, const bool normalize);

// Pairing function to map to a pair of Labels to a single label.
Label pairing(const Label a, const Label b);
Label pairing(const vector<Label> labels);
} // namespace AuxiliaryMethods

#endif // WLFAST_AUXILIARYMETHODS_H