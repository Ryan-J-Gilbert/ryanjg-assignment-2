import numpy as np
import plotly.graph_objects as go

class KMeans():
    def __init__(self, data, n_clusters, init_method='random'):
        self.data = data
        self.k = n_clusters
        self.init_method = init_method
        self.assignment = [-1 for _ in range(len(data))]
        self.snaps = []
        self.centers = None
        self.new_centers = None

    def dist(self, x, y):
        # Euclidean distance
        return np.linalg.norm(x - y)

    def isunassigned(self, i):
        return self.assignment[i] == -1

    def initialize(self, typ='random', manual_centers=None):
        if typ == 'random':
            return self.data[np.random.choice(len(self.data), size=self.k, replace=False)]
        elif typ == 'farthest':
            centers = [self.data[0]]
            for i in range(self.k - 1):
                farthest = 0
                dist = 0
                for j in range(len(self.data)):
                    if self.isunassigned(j):
                        new_dist = sum([self.dist(self.data[j], center) for center in centers])
                        if new_dist > dist:
                            farthest = j
                            dist = new_dist
                centers.append(self.data[farthest])
            return np.array(centers)
        elif typ == 'kmeans++':
            centers = [self.data[np.random.choice(len(self.data))]]
            for i in range(self.k - 1):
                dists = [min([self.dist(center, x) for center in centers]) for x in self.data]
                probs = np.array([d**2 for d in dists]) / sum(d**2 for d in dists)
                centers.append(self.data[np.random.choice(len(self.data), p=probs)])
            return np.array(centers)
        elif typ == 'manual':
            return np.array(manual_centers)
        else:
            raise ValueError("Invalid initialization type")

    def make_clusters(self, centers):
        for i in range(len(self.assignment)):
            for j in range(self.k): 
            # for j in range(len(centers)):
                if self.isunassigned(i):
                    self.assignment[i] = j
                    dist = self.dist(centers[j], self.data[i])
                else:
                    new_dist = self.dist(centers[j], self.data[i])
                    if new_dist < dist:
                        self.assignment[i] = j
                        dist = new_dist

    def compute_centers(self):
        centers = []
        for i in range(self.k):
            cluster = [self.data[j] for j in range(len(self.assignment)) if self.assignment[j] == i]
            if len(cluster) > 0:
                centers.append(np.mean(cluster, axis=0))
            else:
                centers.append(self.data[np.random.choice(len(self.data))])
        return np.array(centers)

    def unassign(self):
        self.assignment = [-1 for _ in range(len(self.data))]

    def are_diff(self, centers, new_centers):
        return any(self.dist(centers[i], new_centers[i]) != 0 for i in range(self.k))

    def snap(self, centers):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.data[:, 0], 
            y=self.data[:, 1], 
            mode='markers', 
            marker=dict(color=self.assignment),
            name='Data Points'
        ))
        fig.add_trace(go.Scatter(
            x=centers[:, 0], 
            y=centers[:, 1], 
            mode='markers', 
            marker=dict(color='red', symbol='cross', size=10),
            name='Centroids'
        ))
        self.snaps.append(fig.to_html(full_html=False))

    def lloyds(self, typ='random', manual_centers=None):
        self.centers = self.initialize(typ, manual_centers)
        self.make_clusters(self.centers)
        self.new_centers = self.compute_centers()
        self.snap(self.new_centers)

    def step(self):
        if self.are_diff(self.centers, self.new_centers):
            self.unassign()
            self.centers = self.new_centers
            self.make_clusters(self.centers)
            self.new_centers = self.compute_centers()
            self.snap(self.new_centers)
            return True
        return False

    def fit(self, data, typ='random', manual_centers=None):
        self.data = data
        self.assignment = [-1 for _ in range(len(data))]
        self.lloyds(typ, manual_centers)

    def plot(self):
        return self.snaps[-1]

    def plot_steps(self):
        return self.snaps
