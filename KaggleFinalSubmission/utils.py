import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot_summaries(history, plot_name1, plot_name2):
    # Plot loss and Accuracy 
    fig, ax = plt.subplots(1, 2, figsize = (16, 8))
    df = pd.DataFrame(history)

    ax[0].plot(df[['root_loss','vowel_loss','consonant_loss', 'val_root_loss','val_vowel_loss','val_consonant_loss']])
    ax[0].set_ylim(0, 2)
    ax[0].set_title('Loss')
    ax[0].legend(['train_root_loss','train_vowel_loss','train_conso_loss', 'val_root_loss','val_vowel_loss','val_conso_loss'],
                loc='upper right')
    ax[0].grid()
    ax[1].plot(df[['root_accuracy','vowel_accuracy','consonant_accuracy', 'val_root_accuracy','val_vowel_accuracy','val_consonant_accuracy']])
    ax[1].set_ylim(0.5, 1)
    ax[1].set_title('Accuracy')
    ax[1].legend(['train_root_acc','train_vowel_acc','train_conso_acc', 'val_root_acc','val_vowel_acc','val_conso_acc'],
                loc='lower right')
    ax[1].grid()
    fig.savefig(plot_name1)

    # Plot Recall 
    fig, ax = plt.subplots(1, 1, figsize = (16, 16))
    df['total_recall'] = (2 * df['root_recall'] + df['vowel_recall_1'] + df['consonant_recall_2']) / 4
    df['val_total_recall'] = (2 * df['val_root_recall'] + df['val_vowel_recall_1'] + df['val_consonant_recall_2']) / 4
    ax.plot(df[['total_recall', 'root_recall','vowel_recall_1','consonant_recall_2', 'val_total_recall', 'val_root_recall','val_vowel_recall_1','val_consonant_recall_2']])
    ax.set_ylim(0.5, 1)
    ax.set_title('Recall')
    ax.legend(['total_recall', 'train_root_recall', 'train_vowel_recall', 'train_conso_recall', 'val_total_recall', 'val_root_recall','val_vowel_recall','val_conso_recall'],
                loc='lower right')
    ax.grid()
    fig.savefig(plot_name2)
