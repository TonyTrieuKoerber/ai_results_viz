import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
from data_utils import *


def get_y_true_and_y_pred_for_cf_matrix(scores: pd.DataFrame, truth: pd.Series, import_params):
    def get_max_index(row):
        return row.idxmax()

    max_column = scores.apply(get_max_index, axis=1)
    y_pred = max_column.apply(lambda x: import_params.model_classes_to_index[x])
    y_true = truth.apply(lambda x: import_params.model_classes_to_index[x])
    return y_true, y_pred

def get_y_true_roc_and_y_good_score_for_roc_curve(df, import_params):
    def convert_y_true_to_y_true_roc(row):
        # returns 0 or 1 depending on row value and positive_index.
        # positive_index can only be 0 or 1 
        if not row == import_params.positive_class:
            return 1 - import_params.pos_label
        else:
            return import_params.pos_label

    y_true_roc = df.truth.apply(convert_y_true_to_y_true_roc)
    y_good_score = df[import_params.positive_class]
    return y_true_roc, y_good_score

def calculate_cf_matrix(y_true:pd.Series, y_pred:pd.Series, import_params) -> np.ndarray:
    cf_matrix_labels = [import_params.model_classes_to_index[x] for x in import_params.model_classes]
    cf_matrix = confusion_matrix(y_true, y_pred, labels=cf_matrix_labels)
    return cf_matrix

def plot_cf_matrix(cf_matrix: np.ndarray, report_folder:str, classes:list):
    "plot and save confusion matrix"
    ax = sns.heatmap(cf_matrix, annot=True, fmt='.5g', cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground Truth')
    ax.xaxis.set_ticklabels(classes, rotation=30)
    ax.yaxis.set_ticklabels(classes, rotation=30)
    plt.savefig(report_folder)

def plot_ovr_frr_dr(y_t: list, y_s: list, save_path: str, n = 5, pos_label=1, benchmark = [0,0]):
    """Takes ground truth and scores for good class and plots complementary ROC curve with
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
    fpr, tpr, thresholds = roc_curve(y_t, y_s, pos_label=pos_label)
    frr = 100*(1 - tpr[::n])
    dr = 100*(1 - fpr[::n])
    thresholds = thresholds[::n]

    plt.figure()
    plt.plot(frr, dr, 'o-', color='darkorange', lw=1, label='threshold')
    plt.plot([0, 100], [0, 100], color='navy', lw=1, linestyle='--')
    plt.plot(*benchmark, 'go', label='vision detection')
    plt.xlim([0.0, 100])
    plt.ylim([0.0, 105])
    plt.xlabel('False Reject Rate [%]')
    plt.ylabel('Detection Rate [%]')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    for x, y, txt in zip(frr,dr,thresholds):
        plt.annotate(np.round(txt,3), (x, y-0.04), rotation = -30, verticalalignment ='top')
    plt.savefig(save_path)


if __name__ == '__main__':
    df = pd.read_csv('./ai_pipeline_export/data/image_based_scores.csv')
    params_path = './ai_pipeline_export/params_C61.yml'
    import_params = load_import_params(params_path, 'import_params')
    get_y_true_and_y_pred_image_based_cf_matrix(df, import_params)


