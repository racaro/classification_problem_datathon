from sklearn.model_selection import train_test_split


class DataSplitter:
    @staticmethod
    def split_data(df, target_column, test_size=0.2, random_state=42):
        """Divide el dataset en conjuntos de entrenamiento y prueba."""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print("Tamaño del conjunto de entrenamiento:", X_train.shape)
        print("Tamaño del conjunto de prueba:", X_test.shape)
        return X_train, X_test, y_train, y_test