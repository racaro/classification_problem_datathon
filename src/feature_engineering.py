from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing import DataPreprocessor


class FeatureEngineer:
    def __init__(self, n_components=2):
        self.pca = PCA(n_components=n_components)

    @staticmethod
    def correlation_matrix(df):
        """Correlation matrix of the dataset."""
        corr = df.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            corr,
            vmin = -1,
            vmax = 1,
            annot = True,
            cbar = True
        )        
        plt.title("Correlation Matrix")
        plt.show()

    @staticmethod
    def apply_pca(X, n_components=3):
        """Appply PCA to the dataset."""
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        return X_pca

    def apply_KMeans(X, n_clusters=2, random_state=42):
        """Apply KMeans to the dataset."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans.fit(X)
        train_clusters = kmeans.predict(X)
        return train_clusters
    