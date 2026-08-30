import joblib 


def save_model(model, filename): 
    """
    Save the trained model to a file using joblib.

    Parameters:
    model: The trained model to be saved.
    filename: The name of the file where the model will be saved.
    """
    joblib.dump(model, filename) 
    
    return f"Model saved to {filename}" 