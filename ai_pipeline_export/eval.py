import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve


def calculate_cf_matrix(y_true:pd.Series, y_pred:pd.Series, import_params) -> np.ndarray:
    cf_matrix_labels = [import_params.prediction_categories_to_index[x] for x in import_params.prediction_categories]
    cf_matrix = confusion_matrix(y_true, y_pred, labels=cf_matrix_labels)
    return cf_matrix

def plot_cf_matrix(cf_matrix: np.ndarray, report_folder:str, classes:list):
    "plot and save confusion matrix"
    ax = sns.heatmap(cf_matrix, annot=True, fmt='.5g', cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground Truth')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)
    plt.savefig(report_folder)

def plot_negative_roc(y_t: list, y_s: list, save_path: str, n = 5, negative_label=1):
    """Takes ground truth and scores for bad class and plots ROC curve with
    detection rate vs false reject rate instead of true positive rate vs false
    positive rate. Only every nth threshold is shown in plot.
    Only for binary classification.

    Args:
        y_t (list): ground truth data points
        y_s (list): scores calculated by the model
        save_path (str): path to save the roc curve
        n (int): only every nth threshold is plotted
        negative_label (int, optional): negative label of ground truth. Defaults to 1.
    """
    frr, dr, thresholds = roc_curve(y_t, y_s, pos_label=negative_label)
    frr = frr[::n]*100
    dr = dr[::n]*100
    thresholds = thresholds[::n]

    plt.figure()
    plt.plot(frr, dr, 'o-', color='darkorange', lw=1, label='threshold')
    plt.plot([0, 100], [0, 100], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 100])
    plt.ylim([0.0, 105])
    plt.xlabel('False Reject Rate [%]')
    plt.ylabel('Detection Rate [%]')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    for x, y, txt in zip(frr,dr,thresholds):
        plt.annotate(np.round(txt,3), (x, y-0.04), rotation = -30, verticalalignment ='top')
    plt.savefig(save_path)