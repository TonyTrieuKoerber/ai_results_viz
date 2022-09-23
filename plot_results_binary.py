import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from pathlib import Path

import csv_utils


def plot_negative_roc(y_t: list, y_s: list, save_path: str, n = 5, negative_label=1):
    """Takes ground truth and scores for bad class and plots ROC curve with
    detection rate vs false reject rate instead of true positive rate vs false
    positive rate. Only every nth threshold is shown in plot.
    Only for binary classification.

    Args:
        y_t (list): ground truth data points
        y_s (list): scores calculated by the model
        n (int): only every nth threshold is plotted
        save_path (str): path to save the roc curve
        negative_label (int, optional): negative label of ground truth. Defaults to 1.
    """
    frr, dr, thresholds = roc_curve(y_t, y_s, pos_label=negative_label)
    frr = frr[::n]*100
    dr = dr[::n]*100
    thresholds = thresholds[::n]

    plt.figure()
    plt.plot(frr, dr, 'o-', color='darkorange', lw=1, label='threshold')
    plt.plot([0, 100], [0, 100], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 10])
    plt.ylim([0.0, 105])
    plt.xlabel('False Reject Rate [%]')
    plt.ylabel('Detection Rate [%]')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    for x, y, txt in zip(frr,dr,thresholds):
        plt.annotate(np.round(txt,3), (x, y-0.04), rotation = -30, verticalalignment ='top')
    plt.savefig(save_path)
    plt.clf()

def calculate_cf_matrix(y_true:pd.Series, y_pred:pd.Series) -> np.ndarray:
    "calculate confusion matrix: input: y_true, y_pred, return: confusion matrix"
    cf_matrix = confusion_matrix(y_true, y_pred)
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
    plt.clf()

def create_predictions_from_threshold(scr:list, th:float, clss: list, lbls:dict) -> list:
    """calculates predictions from scores and labels

    Args:
        scr (list): scores
        th (float): threshold parameter used in prediction
        clss (list): list of classes (e.g. 'good', 'bad')
        lbls (dict): labels used for above mentioned classes

    Returns:
        list: list of predictions
    """
    return list(map(lambda x: label_dict[classes[1]]  if x >= th else label_dict[classes[0]], scr))

def make_directories_and_get_paths(path):
    p = Path(path)
    save_folder = p.parent.parent / '1_statistics'
    if not save_folder.is_dir():
        save_folder.mkdir()
    save_path_roc_png = save_folder / 'roc.png'
    save_path_cf_mat = save_folder / 'cf_matrix.png'
    return save_folder, save_path_roc_png, save_path_cf_mat

def get_sample_truth(sample_name):
    if sample_name in bad_samples:
        return 1
    elif sample_name in good_samples:
        return 0
    else:
        raise ValueError('Sample name not found in "bad_samples" or "good_samples"! Check if file is properly named.')        

def convert_test_scores_to_sample_scores(df_results):
    df_samples = df_results.copy()
    split_image_names = df_results.img.apply(lambda x: Path(x).name.split('_'))
    df_samples['sample'] = split_image_names.apply(lambda x: (x[0],int(x[1])))
    df_samples['rotation'] = split_image_names.apply(lambda x: int(x[2]))
    df_samples['trigger'] = split_image_names.apply(lambda x: int(x[3].split('.')[0]))
    df_samples['sample_truth'] = df_samples['sample'].apply(lambda x: get_sample_truth(x))
    grouped_df = df_samples.groupby(['sample','rotation'])
    predictions = grouped_df['bad'].max()
    sample_truths = grouped_df['sample_truth'].first()
    max_sample_scores = pd.concat([predictions, sample_truths], axis=1)
    # max_sample_scores.to_csv(path_to_scores.parent / "sample_predictions.csv")
    return max_sample_scores


def main(_threshold, score_path: str, save_path_roc: str, save_path_cf_matrix: str, clss:list, neg: str, lbls:dict):
    results = csv_utils.convert_test_scores_csv_to_df(score_path)
    y_scores = results.bad.to_list()
    y_true = results.truth.to_list()

    plot_negative_roc(y_true, y_scores, save_path_roc, n= 5, negative_label=label_dict[neg])
    y_pred = create_predictions_from_threshold(y_scores, _threshold, clss, lbls)
    cf_matrix = calculate_cf_matrix(y_true, y_pred)
    plot_cf_matrix(cf_matrix, save_path_cf_matrix, clss)
    
    sample_results = convert_test_scores_to_sample_scores(results)
    y_scores = sample_results.bad.to_list()
    y_true = sample_results.sample_truth.to_list()
    plot_negative_roc(y_true, y_scores, save_path_roc, n= 5, negative_label=label_dict[neg])
    y_pred = create_predictions_from_threshold(y_scores, _threshold, clss, lbls)
    cf_matrix = calculate_cf_matrix(y_true, y_pred)
    plot_cf_matrix(cf_matrix, save_path_cf_matrix, clss)


if __name__ == '__main__':
    negative_class = 'bad'
    classes = ['good', 'bad']
    # classes = ['good',]
    label_dict = {'good':0, 'bad':1}
    path_to_scores = Path('/home/tonytrieu/repositories/ai_results/Testing/model/test_scores.csv')
    save_folder, save_path_roc_png, save_path_cf_mat = make_directories_and_get_paths(path_to_scores)
    threshold = 0.5

    bad_samples = (
        ('Fail-Cap1', 9,),
        ('Fail-Cap1', 11,),
        ('Fail-Cap1', 14,),
        ('Stopfen', 0),
        )
    good_samples = (
        ('Fail-Cap1', 5,),
        ('Fail-Cap1', 13,),
        ('Fail-Cap1', 15,),
        ('Fail-Neck', 2,),
        ('Good-Neck1', 4),
        ('Good-Neck1', 5),
        ('Good-Neck1', 6),
        )

    main(threshold, path_to_scores, save_path_roc_png, save_path_cf_mat, classes, negative_class, label_dict)
    

