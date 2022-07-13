from locale import normalize
import numpy as np
from copy import deepcopy
from collections import Counter
from sklearn import preprocessing
import threading
from concurrent.futures import ThreadPoolExecutor

# Weisfeiler-Lehman
class Weisfeiler_Lehman:
    def __init__(self, graphs, h):
        self.n = len(graphs)
        self.graphs = graphs
        self.h = h
        self.labels = self.retrieve_all_starting_labels()
        self.labels = {
            index: [str(int(degree)) for degree in self.labels[index].ravel()]
            for index in self.labels
        }
        self.original_labels = deepcopy(self.labels)
        self.compressed_index = int(self.retrieve_highest_degree()[0]) + 1
        self.compressed_labels = {}
        self.count_labels = {}  # {iter1 : {graph1 : count1, graph2 : count2, ...}, ...}
        for i in range(self.n):
            self.count_labels[i] = {}
            self.count_labels[i][0] = self.counter_original_labels(i)
        self.pairwise_similarity_matrix = np.zeros((self.n, self.n))
        del self.original_labels

    def get_graph_starting_labels(self, graph):
        """
        @TODO: Da capire come mixare get_graph_starting_labels dentro retrieve_all_starting_labels.
        Get the starting labels for all the graphs (node degrees), retrieving each graph's
        starting labels.
        {index of the graph in the graphs list : array representing starting label}
        """
        return np.dot(graph, np.ones((len(graph), 1)))

    def retrieve_all_starting_labels(self):
        """
        Get the starting labels for all the graphs (node degrees)
        {index of the graph in the graphs list : array representing starting label}
        """
        starting_labels = {
            index_graph: self.get_graph_starting_labels(self.graphs[index_graph])
            for index_graph in range(self.n)
        }
        return starting_labels

    # def retrieve_all_starting_labels(self):
    #     starting_labels = {
    #         self: np.dot(self.graph[index_graph], np.ones((len(self.graph[index_graph]), 1)))
    #         for index_graph in range(self.n)
    #     }
    #     print(starting_labels)
    #     return starting_labels

    def retrieve_highest_degree(self):
        """
        Retrieve the highest degree of a node in all graphs
        """
        return max([max(v) for _, v in self.retrieve_all_starting_labels().items()])

    def get_neighbours_node(self, index_graph, node):
        """
        Get the neighbours of a node
        index_graph --> index of the graph
        node --> index of the node
        neighbours --> list with indices of the neighbors
        """
        graph = self.graphs[index_graph]
        neighbors = [j for j in range(len(graph)) if graph[node][j] == 1]
        return neighbors

    def get_labels(self, index_graph, node):
        """
        Get updated labels for a node as per (1)
        index_graph --> index of the graph
        node --> index of the node in the graph

        Take the labels of the neighbors of a node, sort them, merge them into an unique string
        """
        new_label = sorted(
            [
                self.labels[index_graph][i]
                for i in self.get_neighbours_node(index_graph, node)
            ]
        )
        new_label = "".join(str(int(i)) for i in new_label)
        return new_label

    def determine_labels(self, index_graph):
        """
        Compute the new multiset of labels of each node in a graph.
        Return a dictionary in which the key is the index of the node and the value
        is the string returned from the `get_labels` function
        index_graph --> index of the graph in the graphs array
        """
        new_labels = {
            l: self.get_labels(index_graph, l)
            for l in range(len(self.graphs[index_graph]))
        }
        return new_labels

    def extend_labels(self, index_graph, new_labels):
        """
        Return the string obtained from the sorted multiset
        index_graph --> index of the graph in the array
        new_labels --> is the array of labels ???????
        """
        for l in new_labels:  # new_labels is a dict
            new_labels[l] = self.labels[index_graph][l] + new_labels[l]
        return new_labels

    def compress_label(self, label):
        """
        Compress a label if it has not been compressed already
        {long_label : compressed_index}
        """
        if label not in self.compressed_labels:
            self.compressed_labels[label] = str(self.compressed_index)
            self.compressed_index += 1
        return self.compressed_labels[label]

    def relabel_nodes(self, index_graph, new_labels):
        """
        Relabel all the nodes in a graph
        """
        assert len(new_labels) == len(self.labels[index_graph])
        for i in range(len(new_labels)):
            self.labels[index_graph][i] = self.compress_label(new_labels[i])

    def counter_original_labels(self, index_graph):
        """
        Count the original node labels: return a list with the number of occurrences per each label
        index_graph --> index of the graph
        [0, 1, 2, 3, 1] --> 0 nodes with label 0, 1 node with label 1, ..., 1 node with label 4
        """
        phi = []
        ol = list(map(int, self.original_labels[index_graph]))
        c = Counter(ol)
        phi = np.zeros(max(ol) + 1)
        for k in range(max(ol) + 1):
            if k in c:
                phi[k] = c[k]
        return phi

    def count_node_label_actual_iteration(self, index_graph):
        """
        Count node labels at current iteration: return a list with the number of occurrences per each label
        index_graph --> index of the graph
        """
        # l = list(map(int, [i for _, i in self.compressed_labels.items()])) non credo venga usata da nessuna parte
        c = Counter(self.labels[index_graph])
        m = max(int(i) for i in c)
        phi = np.zeros(m + 1)
        for k in range(m + 1):
            if str(k) in c:
                phi[k] = c[str(k)]
        return phi

    def set_feature_vectors(self, index_graph_1, index_graph_2):
        """
        Prepare feature vectors for the dot product. Make the shorter ones as long as the long ones,
        concatenate, and so on.
        index_graph_1, index_graph_2 --> indices of two graphs
        """
        l1 = self.count_labels[index_graph_1]
        l2 = self.count_labels[index_graph_2]
        tot1 = []
        tot2 = []
        for i in range(self.h + 1):
            l = min(len(l1[i]), len(l2[i]))
            a1 = l1[i][:l]
            a2 = l2[i][:l]
            tot1 = np.concatenate((tot1, a1), axis=0)
            tot2 = np.concatenate((tot2, a2), axis=0)
        return tot1, tot2

    def normalize_similarity_matrix(self):
        """
        Normalize similarity matrix: sum per rows must be = 1
        """
        self.pairwise_similarity_matrix = preprocessing.normalize(
            self.pairwise_similarity_matrix
        )
        return self.pairwise_similarity_matrix

    def pairwise_similarities(self):
        """
        Pairwise similarities between all the not normalised graphs
        """
        for i in range(self.n):
            for j in range(i, self.n):
                index_graph_1, index_graph_2 = self.set_feature_vectors(i, j)
                dot_product = np.dot(index_graph_1, index_graph_2)
                self.pairwise_similarity_matrix[i][j] = dot_product
                self.pairwise_similarity_matrix[j][i] = dot_product
        self.normalize_similarity_matrix()
        return self.pairwise_similarity_matrix

    def weisfeiler_lehman_algorithm(self):
        """
        The function runs the all the steps of the wiesfeiler-lehman algorithm
        """
        for i in range(self.h):
            for index_graph in range(
                self.n
            ):  # index_graph is the index of the graph in the array of graphs
                new_labels = self.determine_labels(index_graph)
                new_labels = self.extend_labels(index_graph, new_labels)
                self.relabel_nodes(index_graph, new_labels)
                self.count_labels[index_graph][
                    i + 1
                ] = self.count_node_label_actual_iteration(index_graph)
        return self.pairwise_similarities()


# @staticmethod
# def static_weisfeiler_lehman_algorithm(data):
#     for i in range(data.h):
#         for index_graph in range(
#             data.n
#         ):  # index_graph is the index of the graph in the array of graphs
#             new_labels = data.determine_labels(index_graph)
#             new_labels = data.extend_labels(index_graph, new_labels)
#             data.relabel_nodes(index_graph, new_labels)
#             data.count_labels[index_graph][
#                 i + 1
#             ] = data.count_node_label_actual_iteration(index_graph)
#     return data.pairwise_similarities()

# @staticmethod
# def weisfeiler_lehman_algorithm_multi_data(datasets):
#     thread_results = []
#     with ThreadPoolExecutor() as thread_pool:
#         thread_results = thread_pool.map(
#             Weisfeiler_Lehman.static_weisfeiler_lehman_algorithm, datasets
#         )
#     return thread_results
