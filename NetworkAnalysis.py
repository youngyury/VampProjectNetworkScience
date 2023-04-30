import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from parse import VKMutual
from tqdm import tqdm

ACCESS_TOKEN = 'vk1.a.5_VxWz8sMj2EqI4QinoRT_dHMIiYgnsv2qF1qJfVFOL0ArtDGiblzexsO96LR9QImXZrXQhnMOA4' \
               '-cRH0D0SjsnQQL39ZhLwEQLVnqBRrM_oey1Zcf0KGl_KuP68VOZKd8BrKrVVOhj2MAdtVNcFVxSYAHisj1Sx6mz6Tljz' \
               '-vhUBOyu7fuVDJii2gOFFOC_AefxKwhkGvq5iPrJXvyw0Q'
API_V = '5.130'


class Graph:
    """
    Get graph from friends dict
    """
    def __init__(self, dictG):
        self.dictG = dictG
        self.G = self.get_nx_G()

    def get_nx_G(self):
        G = nx.Graph()
        new_dict = {k: v for k, v in self.dictG.items() if len(v) != 0}
        for i, node in enumerate(new_dict):
            G.add_node(node)
            for neighbor in new_dict[node]:
                G.add_edge(node, neighbor)
        return G

    def plot_G(self):
        plt.figure(figsize=(25, 17))
        layout = nx.spring_layout(self.G)
        nx.draw_networkx(self.G, pos=layout, node_size=10)

    def base_info(self):
        print(f'Connected? {nx.is_connected(self.G)}')
        print(f'Radius: {nx.radius(self.G)}')
        print(f'Diameter: {nx.diameter(self.G)}')
        print(f'Average shortest path length: {nx.average_shortest_path_length(self.G)}')
        print(f'Average clustering coefficient: {nx.average_clustering(self.G)}')
        print(f'Number of nodes and edges: {nx.info(self.G)}')

    def avg_path_len(self, n_nodes=np.arange(1, 280, 10)):
        res = {}
        for node in tqdm(self.G.nodes):
            len_ = nx.shortest_path_length(self.G, node)
            res[node] = np.mean(list(len_.values()))
        return res

    def plot_hist_apl(self):
        vals = self.avg_path_len()
        vals = sorted(list(vals.values()))
        plt.bar(x=np.arange(0, len(vals)), height=vals)
        plt.grid()
        plt.title('avg path len')
        plt.xlabel('Nodes')
        plt.show()


class StructAnalysis(Graph):
    def __init__(self, dictG):
        super().__init__(dictG)
        self.G = self.get_nx_G()
        self.pos = nx.kamada_kawai_layout(self.G)

    def random_graphs(self):
        """
        Compare random graphs(er, ba, ws) with ego-graph of friends

        :return: pandas dataframe
        """
        n_nodes = len(self.G.nodes())
        avg_edges = int(np.array([d for n, d in self.G.degree()]).mean())
        prob = avg_edges / n_nodes
        er = nx.fast_gnp_random_graph(n_nodes, prob, seed=1)
        ba = nx.barabasi_albert_graph(n_nodes, avg_edges, seed=1)
        ws = nx.watts_strogatz_graph(n_nodes, avg_edges, p=0.1, seed=1)
        graphs_list = [self.G, er, ba, ws]

        compare_df = pd.DataFrame(data={'clustering': [nx.average_clustering(graph) for graph in graphs_list],
                                        'avg path len': [nx.average_shortest_path_length(graph) for graph in
                                                         graphs_list],
                                        'diameter': [nx.diameter(graph) for graph in graphs_list],
                                        'radius': [nx.radius(graph) for graph in graphs_list]},
                                  index=['my graph', 'erdos-renyi', 'barabasi-albert', 'watts-strogatz'])

        return compare_df

    def centralities(self):
        return {
            "degree": np.array(list(nx.degree_centrality(self.G).values())),
            "closeness": np.array(list(nx.closeness_centrality(self.G).values())),
            "betweenness": np.array(list(nx.betweenness_centrality(self.G).values()))
        }

    def top10_centralities(self):
        r = {}
        for n, f in zip(["degree", "betweenness", "closeness"],
                        [nx.degree_centrality, nx.closeness_centrality, nx.betweenness_centrality]):
            sort = sorted(list(f(self.G).items()), key=lambda x: x[1], reverse=True)
            ids, centrality = zip(*sort)
            friend_id = [y for y in ids]
            r[n] = pd.DataFrame({"id": friend_id, "centrality": centrality})
        return r

    def plot_centralities(self):
        c = self.centralities()
        top_dfs = self.top10_centralities()
        cases = [['degree', 5000, 0, 'Degree centrality'],
                 ['closeness', 700, 0, 'Closeness centrality'],
                 ['betweenness', 900, 20, 'Betweenness centrality']]
        for c_key, scale, bias, title in cases:
            print(title)
            display(top_dfs[c_key])
            plt.figure(figsize=(10, 8))
            nx.draw(self.G,
                    self.pos,
                    width=0.5,
                    linewidths=0.5,
                    edgecolors='black',
                    cmap=plt.cm.hot,
                    node_size=c[c_key] * scale + bias,
                    node_color=c[c_key])
            plt.show()

    def katz_centrality(self, n=1, beta=1 / 30):
        A = nx.to_numpy_array(self.G)
        if beta >= 1 / max(np.linalg.eigvals(A)):
            raise Exception('Error')

        k_centrality = np.zeros(len(A))

        for i in range(1, n + 1):
            k_centrality += beta ** i * np.linalg.matrix_power(A, i).sum(axis=1)

        return k_centrality

    def plot_katz_centrality(self):
        k_centrality = self.katz_centrality()
        k_centrality = k_centrality / k_centrality.max()
        plt.figure(figsize=(8, 6))
        nx.draw(self.G,
                self.pos,
                width=0.5,
                linewidths=0.5,
                edgecolors='black',
                cmap=plt.cm.hot,
                node_size=k_centrality * 300,
                node_color=k_centrality)
        plt.show()

    def eigenvector_centrality(self):
        A = nx.to_numpy_array(self.G)
        w, v = np.linalg.eig(A)
        eig_c = abs(v[:, np.argmax(w)])
        return eig_c

    def plot_eigenvector_centrality(self):
        eig_c = self.eigenvector_centrality()
        plt.figure(figsize=(8, 6))
        nx.draw(self.G,
                self.pos,
                width=0.5,
                linewidths=0.5,
                edgecolors='black',
                cmap=plt.cm.hot,
                node_size=eig_c / eig_c.max() * 400,
                node_color=eig_c)
        plt.show()

    def pearson_correlation(self):
        centvals = [
            ('Degree', list(nx.degree_centrality(self.G).values())),
            ('Closeness', list(nx.closeness_centrality(self.G).values())),
            ('Betweenness', list(nx.betweenness_centrality(self.G).values())),
            ('Katz', self.katz_centrality()),
            ('Eigenvector', self.eigenvector_centrality())
        ]
        plt.figure(figsize=(2 * 5, 5 * 5))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        k = 1
        for i in range(len(centvals)):
            for j in range(i + 1, len(centvals)):
                i_label, i_vals = centvals[i]
                j_label, j_vals = centvals[j]
                correlation = np.corrcoef(i_vals, j_vals)[0][1]
                plt.subplot(5, 2, k)
                plt.scatter(i_vals, j_vals, s=5)
                plt.title('Correlation: {:.4f}'.format(correlation))
                plt.xlabel(i_label)
                plt.ylabel(j_label)
                k += 1


class CommunityDetection(Graph):
    def __init__(self, dictG):
        super().__init__(dictG)
        self.G = self.get_nx_G()
        self.lesG = nx.convert_node_labels_to_integers(self.G)
        self.pos = nx.kamada_kawai_layout(self.lesG)

    @staticmethod
    def k_core_decompose(G):
        return np.array(list(nx.core_number(G).values()))

    def k_core_visualization(self):
        plt.figure(figsize=(8 * 2, 8 * 4))
        x_max, y_max = np.array(list(self.pos.values())).max(axis=0)
        x_min, y_min = np.array(list(self.pos.values())).min(axis=0)
        for i in range(8):
            plt.subplot(4, 2, i + 1)
            subG = nx.k_core(self.lesG, i + 1)
            nodes = nx.draw_networkx_nodes(
                subG,
                self.pos,
                cmap=plt.cm.OrRd,
                node_color=self.k_core_decompose(subG),
                node_size=100,
                edgecolors='black'
            )
            nx.draw_networkx_edges(
                subG,
                self.pos,
                alpha=0.3,
                width=1,
                edge_color='black'
            )
            eps = (x_max - x_min) * 0.05
            plt.xlim(x_min - eps, x_max + eps)
            plt.ylim(y_min - eps, y_max + eps)
            plt.legend(*nodes.legend_elements())
            plt.axis('off')
            plt.title('k-shells on {}-core'.format(i + 1))

    @staticmethod
    def color_clique(G, max_cliques):
        start_end_colors = [np.array([255 / 255, 204 / 255, 229 / 255]), np.array([77 / 255, 0, 36 / 255])]
        step_color = [(x - y) / (len(max_cliques) - 1) for x, y in zip(start_end_colors[1], start_end_colors[0])]

        colors_cliques = []
        color = start_end_colors[0]

        for clique in max_cliques:
            clique_color = []
            for node in G.nodes():
                if node in clique:
                    clique_color.append(color)
                else:
                    clique_color.append([1, 1, 1])
            colors_cliques.append(clique_color)
            color = start_end_colors[1]

        return np.array(colors_cliques)

    @staticmethod
    def edge_width(G, max_cliques):
        edges_width = []

        for clique in max_cliques:
            clique_width = []
            for edge in G.edges():
                if edge[0] in clique and edge[1] in clique:
                    clique_width.append(1.5)
                else:
                    clique_width.append(0.5)
            edges_width.append(clique_width)

        return np.array(edges_width)

    def largest_cliques(self):
        cliques_list = list(nx.find_cliques(self.lesG))
        max_cliques = [clique for clique in cliques_list if len(clique) == len(max(cliques_list, key=lambda x: len(x)))]
        colors_cliques = self.color_clique(self.lesG, max_cliques)
        edges_width = self.edge_width(self.lesG, max_cliques)
        return colors_cliques, edges_width

    def clique_visualization(self):
        colors, widths = self.largest_cliques()
        size = np.unique(colors[0], axis=0, return_counts=True)[1][0]
        plt.figure(figsize=(8 * 3, 8 * 6))
        for i in range(colors.shape[0]):
            b_edges = np.array(list(self.lesG.edges))[widths[i] == widths[i].max()]
            plt.subplot(i + 2, 2, i + 1)
            nodes = nx.draw_networkx_nodes(
                self.lesG,
                self.pos,
                node_color=colors[i],
                node_size=120,
                linewidths=1,
                edgecolors='black'
            )
            nx.draw_networkx_edges(
                self.lesG,
                self.pos,
                alpha=0.3,
                width=widths[i].min()
            )
            nx.draw_networkx_edges(
                self.lesG,
                self.pos,
                width=widths[i].max(),
                edgelist=b_edges
            )
            plt.title('Clique of the size {}'.format(size))
            plt.axis('off')
