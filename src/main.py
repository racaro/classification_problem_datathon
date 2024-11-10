from data_preprocessing import DataPreprocessor
from data_loader import DataLoader

# Data loader
data_loader = DataLoader('data/train.csv')
df = data_loader.load_data()
data_validation_loader = DataLoader('data/validation.csv')