import networkx as nx
import numpy as np


class sbm_generator(object):
    def sbm_dense_balanced(self):
        # balanced case:
        sizes = [150, 150, 150, 150]
        probs = [[0.20, 0.05, 0.02, 0.03], [0.05, 0.30, 0.07, 0.02],
                 [0.02, 0.07, 0.30, 0.05], [0.03, 0.02, 0.05, 0.50]]
        sbm_graph = self.generateSBM(sizes, probs)
        print("sizes:", sizes)
        print("probs:", probs)
        return sbm_graph

    def sbm_dense_imbalanced(self):
        # imbalanced case:
        sizes = [80, 150, 300, 800]
        probs = [[0.35, 0.03, 0.05, 0.03], [0.03, 0.35, 0.02, 0.04],
                 [0.05, 0.02, 0.45, 0.02], [0.03, 0.04, 0.02, 0.45]]
        sbm_graph = self.generateSBM(sizes, probs)
        print("sizes:", sizes)
        print("probs:", probs)
        return sbm_graph

    def sbm_sparse_balanced(self):
        # balanced case:
        sizes = [150, 150, 150, 150]
        probs = [[0.035, 0.003, 0.005, 0.003], [0.003, 0.025, 0.002, 0.004],
                 [0.005, 0.002, 0.045, 0.002], [0.003, 0.004, 0.002, 0.035]]
        sbm_graph = self.generateSBM(sizes, probs)
        print("sizes:", sizes)
        print("probs:", probs)
        return sbm_graph

    def sbm_sparse_imbalanced(self):
        # imbalanced case:
        sizes = [80, 150, 300, 800]
        probs = [[0.025, 0.003, 0.005, 0.003], [0.003, 0.035, 0.002, 0.004],
                 [0.005, 0.002, 0.045, 0.002], [0.003, 0.004, 0.002, 0.035]]
        sbm_graph = self.generateSBM(sizes, probs)
        print("sizes:", sizes)
        print("probs:", probs)
        return sbm_graph

    def sbm_dense_overlap(self):
        sizes = [180, 150, 300, 550]
        probs = [[0.45, 0.13, 0.06, 0.03], [0.13, 0.35, 0.16, 0.09],
                 [0.06, 0.16, 0.40, 0.12], [0.03, 0.09, 0.12, 0.25]]
        sbm_graph = self.generateSBM(sizes, probs)
        print("sizes:", sizes)
        print("probs:", probs)
        return sbm_graph

    def sbm_sparse_overlap(self):
        # imbalanced case:
        sizes = [230, 100, 300, 480]
        probs = [[0.15, 0.07, 0.06, 0.06], [0.07, 0.18, 0.07, 0.09],
                 [0.06, 0.07, 0.21, 0.11], [0.06, 0.09, 0.11, 0.16]]
        sbm_graph = self.generateSBM(sizes, probs)
        print("sizes:", sizes)
        print("probs:", probs)
        return sbm_graph

    def generateSBM(self, sizes, probs):
        """
        docstring
        """
        G = nx.stochastic_block_model(sizes, probs, seed=0)
        return G

    def randomSBM(self, clusters=4, seed=0, size_mu=300, size_sigma=300, p_mu=0.3, p_sigma=0.05, r=2):
        np.random.seed(seed)
        # sizes = np.random.randint(100, 1000, size)
        # mu, sigma = 300, 300  # mean and standard deviation
        
        sizes = np.random.normal(size_mu, size_sigma, clusters)
        while (sizes < 0).any():
            seed += 1
            np.random.seed(seed)
            sizes = np.random.normal(size_mu, size_sigma, clusters)
        sizes = np.round(sizes).astype(np.int64)

        # ps = np.random.uniform(0.1, 0.6, size)
        ps = np.random.normal(p_mu, p_sigma, clusters)
        while (ps < 0).any():
            seed += 1
            np.random.seed(seed)
            ps = np.random.normal(p_mu, p_sigma, clusters)
        probs = np.zeros((len(sizes), len(sizes)))
        for idx, p in enumerate(ps):
            for i in range(len(sizes)):
                probs[idx, i] = p / r
                probs[i, idx] = probs[idx, i]

            probs[idx, idx] = p
        print("sizes:", sizes)
        print("probs:", probs)
        G = nx.stochastic_block_model(sizes, probs, seed=0)



        return G, seed
