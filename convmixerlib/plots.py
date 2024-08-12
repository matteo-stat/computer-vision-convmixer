from matplotlib import pyplot as plt
from typing import Dict
import datetime as dt

def save_plot_training_history(history: Dict, output_path: str) -> None:
    """
    save a plot showing training history loss

    Args:
        history (Dict): history dictionary created after running keras method model.fit
        output_path (str): path where the plot image will be saved
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
    else:
        plt.title('Training Loss')
    plt.xlabel('Epoch')  
    plt.ylabel('Loss')
    plt.legend() 
    plt.grid(True)
    timestamp = dt.datetime.now().strftime('%Y%m%d-%H%M')
    plt.savefig(f'{output_path}/{timestamp}-training-history.png')
