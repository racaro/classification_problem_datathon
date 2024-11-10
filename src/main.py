from data_loader import DataLoader
from data_preprocessing import DataPreprocessor
from data_split import DataSplitter
from feature_engineering import FeatureEngineer
from modeling import ModelTrainer

def main():
    # Load data
    print("Loading data...")
    data_loader = DataLoader('data/train.xlsx')
    df = data_loader.load_data()

    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    df_preprocessed = preprocessor.preprocess_data(df)

    # Perform feature engineering
    print("Performing feature engineering...")
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.transform(df_preprocessed)

    # Split data into training and validation sets
    print("Splitting data into training and validation sets...")
    splitter = DataSplitter()
    X_train, X_val, y_train, y_val = splitter.split_data(df_features)

    # Train and evaluate the model
    print("Training and evaluating the model...")
    model_trainer = Model()
    model = model_trainer.train(X_train, y_train)
    model_trainer.evaluate(model, X_val, y_val)

    print("Process completed.")

if __name__ == "__main__":
    main()
