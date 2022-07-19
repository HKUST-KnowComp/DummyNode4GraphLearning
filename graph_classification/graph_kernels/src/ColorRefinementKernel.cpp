#include "ColorRefinementKernel.h"
#include <set>

namespace ColorRefinement
{
ColorRefinementKernel::ColorRefinementKernel(const GraphDatabase &graph_database)
    : m_graph_database(graph_database), m_label_to_index(), m_num_labels(0)
{
}

GramMatrix ColorRefinementKernel::compute_gram_matrix(const uint num_iterations, const bool use_node_labels,
                                                      const bool use_edge_labels, const bool compute_gram,
                                                      const bool wloa)
{
    size_t num_graphs = m_graph_database.size();
    vector<ColorCounter> color_counters;
    color_counters.reserve(num_graphs);

    // Compute labels for each graph in graph database.
    for (Graph &graph : m_graph_database)
    {
        color_counters.push_back(compute_colors(graph, num_iterations, use_node_labels, use_edge_labels).first);
    }

    // Compute feature vectors.
    vector<S> nonzero_compenents;
    ColorCounter c;
    for (Node i = 0; i < num_graphs; ++i)
    {
        for (const auto &j : color_counters[i])
        {
            Label key = j.first;
            uint value = j.second;
            uint index = m_label_to_index.find(key)->second;
            nonzero_compenents.push_back(S(i, index, value));
        }
    }

    // Compute Gram matrix.
    GramMatrix feature_vectors(num_graphs, m_num_labels);
    feature_vectors.setFromTriplets(nonzero_compenents.begin(), nonzero_compenents.end());

    if (wloa)
    {
        MatrixXd gram_matrix = MatrixXd::Zero(num_graphs, num_graphs);

        // Copy rows to sparse vectors for faster component-wise operations.
        Eigen::SparseVector<double> fvs[num_graphs];
        for (uint i = 0; i < num_graphs; ++i)
        {
            fvs[i] = feature_vectors.row(i);
        }

        for (uint i = 0; i < num_graphs; ++i)
        {
            for (uint j = i; j < num_graphs; ++j)
            {
                double tmp = fvs[i].cwiseMin(fvs[j]).sum();
                gram_matrix(i, j) = tmp;
                gram_matrix(j, i) = tmp;
            }
        }

        return gram_matrix.sparseView();
    }

    if (not compute_gram)
    {
        return feature_vectors;
    }
    else
    {
        GramMatrix gram_matrix(num_graphs, num_graphs);
        gram_matrix = feature_vectors * feature_vectors.transpose();

        return gram_matrix;
    }
}

vector<GramMatrix> ColorRefinementKernel::compute_gram_matrices(const uint num_iterations, const bool use_node_labels,
                                                                const bool use_edge_labels, const bool compute_gram,
                                                                const bool wloa)
{
    size_t num_graphs = m_graph_database.size();
    vector<ColorCounter> color_counters;
    color_counters.reserve(num_graphs);
    vector<vector<uint>> color_numbers;
    color_numbers.reserve(num_graphs);
    vector<GramMatrix> gram_matrices;
    gram_matrices.reserve(num_iterations + 1);

    // Compute labels for each graph in graph database.
    for (Graph &graph : m_graph_database)
    {
        auto colors = compute_colors(graph, num_iterations, use_node_labels, use_edge_labels);
        color_counters.push_back(colors.first);
        color_numbers.push_back(colors.second);
    }

    vector<S> nonzero_compenents;
    uint num_labels = 0;
    for (uint h = 0; h < num_iterations + 1; ++h)
    {
        // Compute feature vectors.
        for (Node i = 0; i < num_graphs; ++i)
        {
            auto it = color_counters[i].begin();
            uint new_num_color = color_numbers[i][h];
            if (h > 0)
            {
                for (uint j = 0; j < color_numbers[i][h - 1]; ++j)
                {
                    ++it;
                }
                new_num_color -= color_numbers[i][h - 1];
            }
            while (new_num_color--)
            {
                Label key = it->first;
                uint value = it->second;
                uint index = m_label_to_index.find(key)->second;
                num_labels = num_labels > index + 1 ? num_labels : index + 1;
                nonzero_compenents.push_back(S(i, index, value));
                ++it;
            }
        }

        GramMatrix feature_vectors(num_graphs, num_labels);
        feature_vectors.setFromTriplets(nonzero_compenents.begin(), nonzero_compenents.end());

        // Compute Gram matrix.
        if (wloa)
        {
            MatrixXd gram_matrix = MatrixXd::Zero(num_graphs, num_graphs);

            if (h > 0)
            {
                // Copy rows to sparse vectors for faster component-wise operations.
                Eigen::SparseVector<double> fvs[num_graphs];
                for (uint i = 0; i < num_graphs; ++i)
                {
                    fvs[i] = feature_vectors.row(i);
                }

                for (uint i = 0; i < num_graphs; ++i)
                {
                    for (uint j = i; j < num_graphs; ++j)
                    {
                        double tmp = fvs[i].cwiseMin(fvs[j]).sum();
                        gram_matrix(i, j) = tmp;
                        gram_matrix(j, i) = tmp;
                    }
                }
            }

            gram_matrices.push_back(gram_matrix.sparseView());
        }
        else if (not compute_gram)
        {
            gram_matrices.push_back(feature_vectors);
        }
        else
        {
            gram_matrices.push_back(feature_vectors * feature_vectors.transpose());
        }
    }

    return gram_matrices;
}

pair<ColorCounter, vector<uint>> ColorRefinementKernel::compute_colors(const Graph &g, const uint num_iterations,
                                                                       bool use_node_labels, bool use_edge_labels)
{
    size_t num_nodes = g.get_num_nodes();

    Labels coloring;
    Labels coloring_temp;

    // Assign labels to nodes.
    if (use_node_labels)
    {
        coloring.reserve(num_nodes);
        coloring_temp.reserve(num_nodes);
        coloring = g.get_labels();
        coloring_temp = coloring;
    }
    else
    {
        coloring.resize(num_nodes, 1);
        coloring_temp = coloring;
    }

    EdgeLabels edge_labels;
    if (use_edge_labels)
    {
        edge_labels = g.get_edge_labels();
    }

    ColorCounter color_map;
    for (Node v = 0; v < num_nodes; ++v)
    {
        Label new_color = coloring[v];

        ColorCounter::iterator it(color_map.find(new_color));
        if (it == color_map.end())
        {
            color_map.insert({{new_color, 1}});
            m_label_to_index.insert({{new_color, m_num_labels}});
            m_num_labels++;
        }
        else
        {
            it->second++;
        }
    }

    vector<uint> color_nums;
    color_nums.reserve(num_iterations + 1);
    color_nums.push_back(color_map.size());

    uint h = 1;
    while (h <= num_iterations && color_nums[h - 1] <= MAXNUMCOLOR)
    {
        // Iterate over all nodes.
        for (Node v = 0; v < num_nodes; ++v)
        {
            Labels colors;

            Nodes neighbors(g.get_neighbours(v));
            colors.reserve(neighbors.size() + 1);

            // New color of node v.
            Label new_color;
            if (!use_edge_labels)
            {
                // Get colors of neighbors.
                for (const Node &n : neighbors)
                {
                    colors.push_back(coloring[n]);
                }
                sort(colors.begin(), colors.end());
                colors.push_back(coloring[v]);

                // Compute new label using composition of pairing function of Matthew Szudzik to map two integers to
                // on integer.
                new_color = colors.back();
                colors.pop_back();
                for (const Label &c : colors)
                {
                    new_color = AuxiliaryMethods::pairing(new_color, c);
                }
                coloring_temp[v] = new_color;
            }
            else
            {
                // Get colors of neighbors.
                for (const Node &n : neighbors)
                {
                    const auto it = edge_labels.find(make_tuple(v, n));
                    colors.push_back(AuxiliaryMethods::pairing(coloring[n], it->second));
                    colors.push_back(coloring[n]);
                }
                sort(colors.begin(), colors.end());
                colors.push_back(coloring[v]);

                // Compute new label using composition of pairing function of Matthew Szudzik to map two integers to
                // on integer.
                new_color = colors.back();
                colors.pop_back();
                for (const Label &c : colors)
                {
                    new_color = AuxiliaryMethods::pairing(new_color, c);
                }
                coloring_temp[v] = new_color;
            }

            // Keep track how often "new_label" occurs.
            auto it = color_map.find(new_color);
            if (it == color_map.end())
            {
                color_map.insert({{new_color, 1}});
                m_label_to_index.insert({{new_color, m_num_labels}});
                m_num_labels++;
            }
            else
            {
                it->second++;
            }
        }

        // Remembers previous number of labels
        color_nums.push_back(color_map.size());

        // Assign new colors.
        std::swap(coloring, coloring_temp);
        h++;
    }

    while (h <= num_iterations)
    {
        color_nums.push_back(color_nums[h - 1]);
        h++;
    }

    return std::make_pair(color_map, color_nums);
}

ColorRefinementKernel::~ColorRefinementKernel()
{
}
} // namespace ColorRefinement