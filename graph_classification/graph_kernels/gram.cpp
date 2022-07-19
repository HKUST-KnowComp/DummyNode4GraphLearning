#include <chrono>
#include <cstdio>
#include <iostream>

#include "src/AuxiliaryMethods.h"
#include "src/ColorRefinementKernel.h"
#include "src/GenerateThree.h"
#include "src/GenerateTwo.h"
#include "src/Graph.h"
#include "src/GraphletKernel.h"
#include "src/ShortestPathKernel.h"

using namespace std::chrono;
using namespace GraphLibrary;
using namespace std;

unordered_map<string, tuple<string, bool, bool>> all_datasets = {
    {"ENZYMES", make_tuple("ENZYMES", true, false)},
    {"DD", make_tuple("DD", true, false)},
    // {"IMDB-BINARY", make_tuple("IMDB-BINARY", false, false)},
    // {"IMDB-MULTI", make_tuple("IMDB-MULTI", false, false)},
    {"IMDB-BINARY", make_tuple("IMDB-BINARY", true, false)},
    {"IMDB-MULTI", make_tuple("IMDB-MULTI", true, false)},
    {"MUTAG", make_tuple("MUTAG", true, true)},
    {"NCI1", make_tuple("NCI1", true, false)},
    {"NCI109", make_tuple("NCI109", true, false)},
    {"PTC_FM", make_tuple("PTC_FM", true, false)},
    {"PTC_FR", make_tuple("PTC_FR", true, false)},
    {"PROTEINS", make_tuple("PROTEINS", true, false)},
    {"REDDIT-BINARY", make_tuple("REDDIT-BINARY", false, false)},
    {"Yeast", make_tuple("Yeast", true, true)},
    {"YeastH", make_tuple("YeastH", true, true)},
    {"UACC257", make_tuple("UACC257", true, true)},
    {"UACC257H", make_tuple("UACC257H", true, true)},
    {"OVCAR-8", make_tuple("OVCAR-8", true, true)},
    {"OVCAR-8H", make_tuple("OVCAR-8H", true, true)}};

int main(int argc, char **argv)
{
    string dataset_dir = "./datasets";
    string gram_dir = "./svm/GM/EXP";
    uint k = 1;
    string kernel = "WL";
    uint n_iters = 1;
    vector<tuple<string, bool, bool>> datasets;

    uint i = 1;
    while (i < argc)
    {
        if (strcmp(argv[i], "--dataset_dir") == 0)
        {
            dataset_dir = string(argv[i + 1]);
            i += 2;
        }
        else if (strcmp(argv[i], "--gram_dir") == 0)
        {
            gram_dir = string(argv[i + 1]);
            i += 2;
        }
        else if (strcmp(argv[i], "--k") == 0)
        {
            k = atoi(argv[i + 1]);
            i += 2;
        }
        else if (strcmp(argv[i], "--kernel") == 0)
        {
            kernel = string(argv[i + 1]);
            i += 2;
        }
        else if (strcmp(argv[i], "--n_iters") == 0)
        {
            n_iters = atoi(argv[i + 1]);
            i += 2;
        }
        else if (strcmp(argv[i], "--datasets") == 0)
        {
            uint j = i + 1;
            while (j < argc)
            {
                if (strlen(argv[j]) > 2 && strncmp(argv[j], "--", 2) == 0)
                {
                    break;
                }
                string dataset = string(argv[j]);
                auto it = all_datasets.find(dataset);
                if (it != all_datasets.end())
                {
                    datasets.push_back(it->second);
                }
                else
                {
                    datasets.push_back(make_tuple(dataset, true, true));
                    cout << "Warning: " << dataset << " is not a valid dataset." << endl;
                }
                ++j;
            }
            i = j;
        }
        else
        {
            cout << "Unknown args: " << argv[i];
            ++i;
        }
    }

    vector<GraphDatabase> gdbs;
    vector<vector<int>> classes;
    for (auto &d : datasets)
    {
        string ds = std::get<0>(d);
        GraphDatabase gdb = AuxiliaryMethods::read_graph_txt_file(dataset_dir, ds);
        gdb.erase(gdb.begin() + 0);
        gdbs.push_back(gdb);
        classes.push_back(AuxiliaryMethods::read_classes(dataset_dir, ds));
    }

    for (uint d = 0; d < datasets.size(); ++d)
    {
        GraphDatabase &gdb = gdbs[d];
        string &ds = std::get<0>(datasets[d]);
        bool use_labels = std::get<1>(datasets[d]);
        bool use_edge_labels = std::get<2>(datasets[d]);

        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        // WL
        if (k == 1)
        {
            if (kernel.compare("WL") == 0)
            {
                ColorRefinement::ColorRefinementKernel graph_kernel(gdb);

                // for (uint i = 0; i <= n_iters; ++i)
                // {
                //     GramMatrix gm = graph_kernel.compute_gram_matrix(i, use_labels, use_edge_labels, true, false);
                //     AuxiliaryMethods::write_libsvm(
                //         gm, classes[d],
                //         gram_dir + "/" + ds + "__" + kernel + to_string(k) + "_" + to_string(i) + ".gram",
                //         true);
                //     gm = GramMatrix(); // release memory
                // }
                vector<GramMatrix> gms =
                    graph_kernel.compute_gram_matrices(n_iters, use_labels, use_edge_labels, true, false);
                for (uint i = 0; i <= n_iters; ++i)
                {
                    AuxiliaryMethods::write_libsvm(
                        gms[i], classes[d],
                        gram_dir + "/" + ds + "__" + kernel + to_string(k) + "_" + to_string(i) + ".gram", true);
                }
                gms.clear();
                gms.shrink_to_fit(); // release memory
            }
            else if (kernel.compare("WLOA") == 0)
            {
                ColorRefinement::ColorRefinementKernel graph_kernel(gdb);

                // for (uint i = 0; i <= n_iters; ++i)
                // {
                //     GramMatrix gm = graph_kernel.compute_gram_matrix(i, use_labels, use_edge_labels, true, true);
                //     AuxiliaryMethods::write_libsvm(
                //         gm, classes[d],
                //         gram_dir + "/" + ds + "__" + kernel + "_" + to_string(i) + ".gram",
                //         true);
                //     gm = GramMatrix(); // release memory
                // }

                vector<GramMatrix> gms =
                    graph_kernel.compute_gram_matrices(n_iters, use_labels, use_edge_labels, true, true);
                for (uint i = 0; i <= n_iters; ++i)
                {
                    AuxiliaryMethods::write_libsvm(
                        gms[i], classes[d], gram_dir + "/" + ds + "__" + kernel + "_" + to_string(i) + ".gram", true);
                }
                gms.clear();
                gms.shrink_to_fit(); // release memory
            }
            else if (kernel.compare("SP") == 0)
            {
                ShortestPathKernel::ShortestPathKernel graph_kernel(gdb);

                GramMatrix gm = graph_kernel.compute_gram_matrix(use_labels, true);
                AuxiliaryMethods::write_libsvm(
                    gm, classes[d], gram_dir + "/" + ds + "__" + kernel + "_" + to_string(0) + ".gram", true);
                gm = GramMatrix(); // release memory
            }
            else if (kernel.compare("GR") == 0)
            {
                GraphletKernel::GraphletKernel graph_kernel(gdb);

                GramMatrix gm = graph_kernel.compute_gram_matrix(use_labels, use_edge_labels, true);
                AuxiliaryMethods::write_libsvm(
                    gm, classes[d], gram_dir + "/" + ds + "__" + kernel + "_" + to_string(0) + ".gram", true);
                gm = GramMatrix(); // release memory
            }
        }
        else if (k == 2)
        {
            if (kernel.find("WL") != string::npos)
            {
                string algorithm;
                if (kernel.compare("WL") == 0)
                {
                    algorithm = "wl";
                }
                else if (kernel.compare("DWL") == 0)
                {
                    algorithm = "malkin";
                }
                else if (kernel.compare("LWL") == 0)
                {
                    algorithm = "local";
                }
                else if (kernel.compare("LWLP") == 0)
                {
                    algorithm = "localp";
                }
                else if (kernel.compare("LWLC") == 0)
                {
                    algorithm = "localc";
                }
                else if (kernel.compare("LWLPC") == 0)
                {
                    algorithm = "localpc";
                }
                else
                {
                    throw std::invalid_argument("Error: unsupported kernel " + kernel);
                }
                GenerateTwo::GenerateTwo graph_kernel(gdb);

                // for (uint i = 0; i <= n_iters; ++i)
                // {
                //     GramMatrix gm = graph_kernel.compute_gram_matrix(i, use_labels, use_edge_labels, algorithm, true,
                //     true); AuxiliaryMethods::write_libsvm(
                //         gm, classes[d],
                //         gram_dir + "/" + ds + "__" + kernel + to_string(k) + "_" + to_string(i) + ".gram",
                //         true);
                //     gm = GramMatrix(); // release memory
                // }

                vector<GramMatrix> gms =
                    graph_kernel.compute_gram_matrices(n_iters, use_labels, use_edge_labels, algorithm, true, true);
                for (uint i = 0; i <= n_iters; ++i)
                {
                    AuxiliaryMethods::write_libsvm(
                        gms[i], classes[d],
                        gram_dir + "/" + ds + "__" + kernel + to_string(k) + "_" + to_string(i) + ".gram", true);
                }
                gms.clear();
                gms.shrink_to_fit(); // release memory
            }
        }
        else if (k == 3)
        {
            if (kernel.find("WL") != string::npos)
            {
                string algorithm;
                if (kernel.compare("WL") == 0)
                {
                    algorithm = "wl";
                }
                else if (kernel.compare("DWL") == 0)
                {
                    algorithm = "malkin";
                }
                else if (kernel.compare("LWL") == 0)
                {
                    algorithm = "local";
                }
                else if (kernel.compare("LWLP") == 0)
                {
                    algorithm = "localp";
                }
                else if (kernel.compare("LWLC") == 0)
                {
                    algorithm = "localc";
                }
                else if (kernel.compare("LWLPC") == 0)
                {
                    algorithm = "localpc";
                }
                else
                {
                    throw std::invalid_argument("Error: unsupported kernel " + kernel);
                }
                GenerateThree::GenerateThree graph_kernel(gdb);

                // for (uint i = 0; i <= n_iters; ++i)
                // {
                //     GramMatrix gm = graph_kernel.compute_gram_matrix(i, use_labels, use_edge_labels, algorithm, true,
                //     true); AuxiliaryMethods::write_libsvm(
                //         gm, classes[d],
                //         gram_dir + "/" + ds + "__" + kernel + to_string(k) + "_" + to_string(i) + ".gram",
                //         true);
                //     gm = GramMatrix(); // release memory
                // }

                vector<GramMatrix> gms =
                    graph_kernel.compute_gram_matrices(n_iters, use_labels, use_edge_labels, algorithm, true, true);
                for (uint i = 0; i <= n_iters; ++i)
                {
                    AuxiliaryMethods::write_libsvm(
                        gms[i], classes[d],
                        gram_dir + "/" + ds + "__" + kernel + to_string(k) + "_" + to_string(i) + ".gram", true);
                }
                gms.clear();
                gms.shrink_to_fit(); // release memory
            }
        }
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(t2 - t1).count();
        cout << kernel + "-" + to_string(n_iters) << "\t" << ds << "\t" << duration << " seconds" << endl;
    }

    return 0;
}
