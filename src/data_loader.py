import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        """Data loader class."""
        self.file_path = file_path

    def load_data(self):
        """Load data from file."""
        return pd.read_csv(self.file_path)
    
    