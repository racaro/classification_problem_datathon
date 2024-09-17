from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.impute import SimpleImputer, KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations


class DataPreprocessor:
    def __init__(self, strategy='median', neighbors=5):
        """Preprocess the dataset."""
        self.imputer = SimpleImputer(strategy=strategy)
        self.imputer_knn = KNNImputer(neighbors=neighbors)
        self.scaler = StandardScaler()
        self.normalizer = Normalizer()


    @staticmethod
    def extract_data_features(df, feature):
        """Extract the features from the dataset."""
        X = df.drop(feature, axis=1)
        y = df[feature]
        return X, y
    
    @staticmethod
    def binary_counter(df_feature):
        """Count the number of zeros or ones in the dataset."""
        zero_counter = len([x for x in df_feature if x == 0])
        one_counter = len([x for x in df_feature if x == 1])
        print(f"There are {zero_counter} 'zeros' and {one_counter} 'ones' in the column {df_feature}.")

    @staticmethod
    def descriptive_statistics(df):  
        """Descriptive statistics of the dataset."""
        stats = df.describe().T["count", "min", "max", "mean", "std"]
        print(stats)

    @staticmethod
    def plot_missing_values(df):
        """Visualiza un mapa de calor de los valores NaN en el dataset."""
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isna(), cbar=False, cmap='viridis')
        plt.title("Mapa de Calor de Valores Ausentes")
        plt.show()
    
    def missing_values_percentage(self, df):
        """Shows the percentage of NaN values per column."""
        na_total = df.isna().sum()
        print(f"There are {na_total.sum()} missing values in the dataset.")
        na_percentage = df.isna().mean() * 100
        print(f"Percentage of missing values per column:\n{na_percentage[na_percentage > 0].sort_values(ascending=False)}")
        self.plot_missing_values(df)

    def check_missing_values_in_dataframe(self, df):
        """Shows the rows with NaN values in the dataset."""
        nan_matrix = df.isna()
        print("Dataframe with NaN values:\n", nan_matrix)
        all_nan_rows = nan_matrix.all(axis=1)
        print("Rows with all NaN values:\n", all_nan_rows)
        any_nan_rows = nan_matrix.any(axis=1)
        print("Rows with any NaN values:\n", any_nan_rows)
        self.plot_missing_values(df)

    @staticmethod
    def scatterplot_features(df):
        """Scatterplot of the features."""
        # Visualization of the features
        columns = df.columns
        # Configurate the size of the figure
        plt.figure(figsize=(14, 16))
        # Iterate over the columns and create subplots
        for i, col in enumerate(columns, 1):
            plt.subplot(4, 2, i)  # subplots create (4 rows, 2 columns)
            sns.scatterplot(data=df, x=df.index, y=col, s=50, color='blue')
            plt.title(f'Scatterplot de {col}')
            plt.xlabel('Índice')
            plt.ylabel(col)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def scatterplot_between_features(df, feature1, feature2):
        """Scatterplot between two features."""
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=feature1,y=feature2)
        plt.title(f'Scatter Plot entre {feature1} y {feature2} para visualizar outliers')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

    def plot_outliers(self, df):
        """Plots the outliers in the dataset."""
        for col1, col2 in combinations(df.columns, 2):
            self.scatterplot_between_features(df, col1, col2)
    
    @staticmethod
    def show_outliers_zscore(data, threshold=3):
        """Detect outliers using z-score."""
        mean = np.mean(data)
        std = np.std(data)
        z_scores = [(x - mean) / std for x in data]
        outliers = [i for i, z in enumerate(z_scores) if np.abs(z) > threshold]
        return outliers
    
    def check_outliers_zscore(self, df):
        outliers_zscore = {col: self.show_outliers_zscore(df[col]) for col in df.columns}
        print("Outliers detectados con Z-score:")
        for col, indices in outliers_zscore.items():
            print(f"{col}: {indices} ({len(indices)} outliers)")

    def impute_knn(self, X):
        """Imputes missing values using KNN."""
        return self.imputer_knn.fit_transform(X)

    def impute(self, X):
        """Imputes missing values in the dataset."""
        return self.imputer.fit_transform(X)
    
    def normalize(self, X):
        """Normalize the data."""
        return self.normalizer.fit_transform(X)

    def standardize(self, X):
        """Standardize the data."""
        return self.scaler.fit_transform(X)
    