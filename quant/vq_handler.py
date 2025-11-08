import torch


def vqHandler(
        apply,          # whether VQ is asked or not (either 1 or 0)
        method,         # which VQ method
        J_vq,           # number of quantization bits
        Q,              # vector dimension, Q=1: scalar quantization
        device=None,    # torch.device('cpu') or torch.device('cuda')
        seed=0          # seed for reproducibility
    ):
    # Returns the vector quantizer if asked
    if apply==1:
        if method=="kmeans++":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
            return KMeans(2**J_vq, Q, device, seed=seed)
        else:
            raise Exception("Invalid vector quantization method")
    else:
        return 0
    

class KMeans:
    def __init__(self, 
            n_clusters,         # number of clusters, M = 2^J
            Q,                  # vector dimension, Q=1: scalar quantization
            device,             # torch.device('cpu') or torch.device('cuda')
            max_iter=300,       # maximum number of iterations for Kmeans++
            seed=0              # seed for reproducibility
        ):
        
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.Q = Q
        self.seed = seed
        self.device = device

    # ---------------------------------------------------------
    def fit(self, X_train):
        # Perform Kmeans++ to learn the centroids with the given data
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        X_train = X_train.to(self.device)

        n_samples, dim = X_train.shape
        self.centroids = torch.zeros((self.n_clusters, dim), device=self.device)

        # Initialize centroids using k-means++ method
        first_centroid_idx = torch.randint(0, n_samples, (1,), device=self.device).item()
        self.centroids[0] = X_train[first_centroid_idx]

        for i in range(1, self.n_clusters):
            dists = torch.min(torch.cdist(X_train, self.centroids[:i], p=2), dim=1)[0]
            dists_prob = dists.pow(2) / torch.sum(dists.pow(2))
            new_centroid_idx = torch.multinomial(dists_prob, 1).item()
            self.centroids[i] = X_train[new_centroid_idx]

        # Run the iterative k-means clustering
        for _ in range(self.max_iter):
            # Compute distances between each point and the centroids
            dists = torch.cdist(X_train, self.centroids, p=2)
            labels = torch.argmin(dists, dim=1)

            # Compute new centroids
            new_centroids = torch.zeros_like(self.centroids)
            for i in range(self.n_clusters):
                cluster_points = X_train[labels == i]
                if cluster_points.shape[0] > 0:  # Update centroid only if cluster is not empty
                    new_centroids[i] = cluster_points.mean(dim=0)

            # Check for convergence
            if torch.max(torch.norm(self.centroids - new_centroids, dim=1)) < 1e-6:
                break

            self.centroids = new_centroids

    # ---------------------------------------------------------
    def evaluate(self, X):
        # Apply quantization
        X = X.to(self.device)
        # Compute distances between each point and centroids
        dists = torch.cdist(X, self.centroids, p=2)
        # Assign each point to the nearest centroid
        labels = torch.argmin(dists, dim=1)
        # Replace each point with its corresponding centroid
        centroids = self.centroids[labels]
        return centroids, labels.tolist()
