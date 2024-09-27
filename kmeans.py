import numpy as np
import plotly.graph_objects as go

class KMeans():
    def __init__(self, data, n_clusters, init_method='random'):
        self.data = data
        self.k = n_clusters
        self.init_method = init_method
        self.assignment = [-1 for _ in range(len(data))]
        self.snaps = []

    def dist(self, x, y):
        # Euclidean distance
        return sum((x - y)**2) ** (1/2)

    def isunassigned(self, i):
        return self.assignment[i] == -1

    def initialize(self, typ='random'):
        if typ == 'random':
            return self.data[np.random.choice(len(self.data) - 1, size=self.k, replace=False)]
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
            centers = [self.data[np.random.choice(len(self.data) - 1)]]
            for i in range(self.k - 1):
                dists = [min([self.dist(center, x) for center in centers]) for x in self.data]
                probs = [dist**2 / sum(dists) for dist in dists]
                probs = np.array(probs) / sum(probs)
                centers.append(self.data[np.random.choice(len(self.data), p=probs)])
            return np.array(centers)
        elif typ == 'Manual':
            raise NotImplementedError()
        else:
            raise ValueError("Invalid initialization type")

    def make_clusters(self, centers):
        for i in range(len(self.assignment)):
            for j in range(self.k):
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
            cluster = []
            for j in range(len(self.assignment)):
                if self.assignment[j] == i:
                    cluster.append(self.data[j])
            centers.append(np.mean(np.array(cluster), axis=0))
        return np.array(centers)

    def unassign(self):
        self.assignment = [-1 for _ in range(len(self.data))]

    def are_diff(self, centers, new_centers):
        for i in range(self.k):
            if self.dist(centers[i], new_centers[i]) != 0:
                return True
        return False

    def snap(self, centers):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data[:, 0], y=self.data[:, 1], mode='markers', marker=dict(color=self.assignment)))
        fig.add_trace(go.Scatter(x=centers[:, 0], y=centers[:, 1], mode='markers', marker=dict(color='red', size=10)))
        self.snaps.append(fig.to_html(full_html=False))

    def lloyds(self, typ='random'):
        self.centers = self.initialize(typ)
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

    def fit(self, data):
        self.data = data
        self.assignment = [-1 for _ in range(len(data))]
        self.lloyds(self.init_method)

    def plot(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data[:, 0], y=self.data[:, 1], mode='markers', marker=dict(color=self.assignment)))
        centers = self.compute_centers()
        fig.add_trace(go.Scatter(x=centers[:, 0], y=centers[:, 1], mode='markers', marker=dict(color='red', size=10)))
        return fig.to_html(full_html=False)


# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image as im

# class KMeans():
#     def __init__(self, data, n_clusters, init_method='random'):
#         self.data = data
#         self.k = n_clusters
#         self.init_method = init_method
#         self.assignment = [-1 for _ in range(len(data))]
#         self.snaps = []

#     def dist(self, x, y):
#         # Euclidean distance
#         return sum((x - y)**2) ** (1/2)

#     def isunassigned(self, i):
#         return self.assignment[i] == -1

#     def initialize(self, typ='random'):
#         if typ == 'random':
#             return self.data[np.random.choice(len(self.data) - 1, size=self.k, replace=False)]
#         elif typ == 'farthest_first':
#             centers = [self.data[0]]
#             for i in range(self.k - 1):
#                 farthest = 0
#                 dist = 0
#                 for j in range(len(self.data)):
#                     if self.isunassigned(j):
#                         new_dist = sum([self.dist(self.data[j], center) for center in centers])
#                         if new_dist > dist:
#                             farthest = j
#                             dist = new_dist
#                 centers.append(self.data[farthest])
#             return np.array(centers)
#         elif typ == 'kmeans++':
#             centers = [self.data[np.random.choice(len(self.data) - 1)]]
#             for i in range(self.k - 1):
#                 dists = [min([self.dist(center, x) for center in centers]) for x in self.data]
#                 probs = [dist**2 / sum(dists) for dist in dists]
#                 probs = np.array(probs) / sum(probs)
#                 centers.append(self.data[np.random.choice(len(self.data), p=probs)])
#             return np.array(centers)
#         elif typ == 'Manual':
#             raise NotImplementedError()
#         else:
#             raise ValueError("Invalid initialization type")

#     def make_clusters(self, centers):
#         for i in range(len(self.assignment)):
#             for j in range(self.k):
#                 if self.isunassigned(i):
#                     self.assignment[i] = j
#                     dist = self.dist(centers[j], self.data[i])
#                 else:
#                     new_dist = self.dist(centers[j], self.data[i])
#                     if new_dist < dist:
#                         self.assignment[i] = j
#                         dist = new_dist

#     def compute_centers(self):
#         centers = []
#         for i in range(self.k):
#             cluster = []
#             for j in range(len(self.assignment)):
#                 if self.assignment[j] == i:
#                     cluster.append(self.data[j])
#             centers.append(np.mean(np.array(cluster), axis=0))
#         return np.array(centers)

#     def unassign(self):
#         self.assignment = [-1 for _ in range(len(self.data))]

#     def are_diff(self, centers, new_centers):
#         for i in range(self.k):
#             if self.dist(centers[i], new_centers[i]) != 0:
#                 return True
#         return False

#     def snap(self, centers):
#         TEMPFILE = "temp.png"
#         fig, ax = plt.subplots()
#         ax.scatter(self.data[:, 0], self.data[:, 1], c=self.assignment)
#         ax.scatter(centers[:, 0], centers[:, 1], c='r')
#         fig.savefig(TEMPFILE)
#         plt.close()
#         self.snaps.append(im.fromarray(np.asarray(im.open(TEMPFILE))))

#     def lloyds(self, typ='random'):
#         centers = self.initialize(typ)
#         self.make_clusters(centers)
#         new_centers = self.compute_centers()
#         self.snap(new_centers)
#         while self.are_diff(centers, new_centers):
#             self.unassign()
#             centers = new_centers
#             self.make_clusters(centers)
#             new_centers = self.compute_centers()
#             self.snap(new_centers)
#         return
    
#     def step(self):
#         if self.are_diff(self.centers, self.new_centers):
#             self.unassign()
#             self.centers = self.new_centers
#             self.make_clusters(self.centers)
#             self.new_centers = self.compute_centers()
#             self.snap(self.new_centers)
#             return True
#         return False

#     def fit(self, data):
#         self.data = data
#         self.assignment = [-1 for _ in range(len(data))]
#         self.lloyds(self.init_method)

#     def plot(self, ax):
#         ax.scatter(self.data[:, 0], self.data[:, 1], c=self.assignment)
#         centers = self.compute_centers()
#         ax.scatter(centers[:, 0], centers[:, 1], c='r')