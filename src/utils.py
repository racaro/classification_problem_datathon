import pickle

def save_model(model, filename):
    """Save a model to a .pkl file."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    """Load a model from a .pkl file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)