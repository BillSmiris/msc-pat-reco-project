import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, isspmatrix_csr
from scipy.sparse.linalg import norm

class KMeansClustering:
    def __init__(self, k=3, distance_metric='euclidean', random_state=None):
        self.k = k
        self.centroids = None
        self.distance_metric = distance_metric
        self.distance_function = self._get_distance_function(distance_metric)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    # Function to calculate the euclidean distance between two user ranking vectors
    def euclidean_dist(R_u, R_v):
        # Extract required vectors from the sparse data and calculate lambdas
        R_u = R_u.toarray().flatten() if isinstance(R_u, csr_matrix) else R_u.flatten()
        R_v = R_v.toarray().flatten() if isinstance(R_v, csr_matrix) else R_v.flatten()
        lambda_u = np.where(R_u > 0, 1, 0)
        lambda_v = np.where(R_v > 0, 1, 0)

        # Check if vectors are valid for comparison
        mask = lambda_u * lambda_v
        if np.sum(mask) == 0:
            return float('inf')

        # Calculate distance with the euclidean formula
        dist = np.sqrt(np.sum(np.square((R_u - R_v) * mask)))
        return dist
    
    @staticmethod
    # Function to calculate the cosine distance between two user ranking vectors
    def cosine_dist(R_u, R_v):
        # Extract required vectors from the sparse data and calculate lambdas
        R_u = R_u.toarray().flatten() if isinstance(R_u, csr_matrix) else R_u.flatten()
        R_v = R_v.toarray().flatten() if isinstance(R_v, csr_matrix) else R_v.flatten()
        lambda_u = np.where(R_u > 0, 1, 0)
        lambda_v = np.where(R_v > 0, 1, 0)

        # Check if vectors are valid for comparison
        mask = lambda_u * lambda_v
        if np.sum(mask) == 0:
            return float('inf')

        # Calculate similarity fraction components
        numerator = np.sum(R_u * R_v * mask)

        denominator_u = np.sqrt(np.sum(np.square(R_u) * mask))
        denominator_v = np.sqrt(np.sum(np.square(R_v) * mask))

        if denominator_u == 0 or denominator_v == 0:
            return float('inf')

        # Calculate distance using the cosine formula
        dist = 1 - np.abs(numerator / (denominator_u * denominator_v))

        return dist

    @staticmethod
    def _get_distance_function(metric):
        if metric == 'euclidean':
            return KMeansClustering.euclidean_dist
        elif metric == 'cosine':
            return KMeansClustering.cosine_dist
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")

    def fit(self, X, max_iterations=100):
        if not isspmatrix_csr(X):
            raise ValueError("X should be a CSR matrix")

        # Initialize centroids with random samples from the data
        n_samples, n_features = X.shape
        random_indices = self.rng.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices].toarray()  # Initial centroids as dense array for computation

        for iterations in range(max_iterations):
            labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                data_point = X.getrow(i).toarray()
                distances = [self.distance_function(data_point, centroid) for centroid in self.centroids]
                labels[i] = np.argmin(distances)

            cluster_centers = np.zeros((self.k, n_features))
            counts = np.zeros(self.k)

            for i in range(n_samples):
                cluster_num = labels[i]
                cluster_centers[cluster_num] += X.getrow(i).toarray().ravel()
                counts[cluster_num] += 1

            for i in range(self.k):
                if counts[i] == 0:
                    # Reinitialize empty clusters
                    cluster_centers[i] = X[np.random.choice(n_samples)].toarray()
                else:
                    cluster_centers[i] /= counts[i]

            if np.linalg.norm(self.centroids - cluster_centers) < 0.0001:
                break
            else:
                self.centroids = cluster_centers
        return labels