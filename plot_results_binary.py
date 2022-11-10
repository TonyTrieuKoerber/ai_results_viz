import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from pathlib import Path
import json

import utils


def load_config(path:str) -> list:
    with open(path) as f:
        config_dict = json.load(f)
    for key in config_dict['sample_dict']:
        config_dict['sample_dict'][key] = list(map(tuple, config_dict['sample_dict'][key]))
    config = (
        Path(config_dict['path_to_scores']), 
        config_dict['negative_class'],
        config_dict['classes'],
        config_dict['class_dict'], 
        config_dict['threshold'],
        config_dict['sample_dict'],
    )
    return config

def make_statistics_dir_and_get_paths(path):
    p = Path(path)
    save_folder = p.parent.parent / '1_statistics'
    if not save_folder.is_dir():
        save_folder.mkdir()
    save_path_roc = save_folder / 'image_based_roc.png'
    save_path_cf_mat = save_folder / 'image_based_cf_matrix.png'
    return save_path_roc, save_path_cf_mat

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

def calculate_cf_matrix(y_true:pd.Series, y_pred:pd.Series, cls_dict: dict, clss: list) -> np.ndarray:
    cf_matrix_labels = [cls_dict[x] for x in clss]
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
    return list(map(lambda x: lbls[clss[1]]  if x >= th else lbls[clss[0]], scr))


def get_sample_truth(sample_name, sam_dict):
    if sample_name in sam_dict['bad']:
        return 1
    elif sample_name in sam_dict['good']:
        return 0
    else:
        raise ValueError(f'"{sample_name}" not found in "bad_samples" or "good_samples"! Check if file is properly named.')        

def convert_test_scores_to_sample_scores(df_results, sam_dict, csv_save_path):
    df_samples = df_results.copy()
    split_image_names = df_results.img.apply(lambda x: Path(x).name.split('_'))
    df_samples['sample'] = split_image_names.apply(lambda x: (x[0],int(x[1])))
    df_samples['rotation'] = split_image_names.apply(lambda x: int(x[2]))
    df_samples['trigger'] = split_image_names.apply(lambda x: int(x[3].split('.')[0]))
    df_samples['sample_truth'] = df_samples['sample'].apply(lambda x: get_sample_truth(x, sam_dict))
    grouped_df = df_samples.groupby(['sample','rotation'])
    predictions = grouped_df['bad'].max()
    sample_truths = grouped_df['sample_truth'].first()
    max_sample_scores = pd.concat([predictions, sample_truths], axis=1)
    max_sample_scores.to_csv(csv_save_path.parent / "sample_predictions.csv")
    return max_sample_scores


def main(score_path: str, _threshold, clss:list, neg: str, cls_dict:dict, sam_dict: dict):
    save_path_roc, save_path_cf_matrix = make_statistics_dir_and_get_paths(score_path)
    test_scores = utils.convert_test_scores_csv_to_df(score_path)
    y_scores = test_scores.bad
    y_true = test_scores.truth

    plot_negative_roc(y_true, y_scores, save_path_roc, n= 5, negative_label=cls_dict[neg])
    y_pred = create_predictions_from_threshold(y_scores, _threshold, clss, cls_dict)
    cf_matrix = calculate_cf_matrix(y_true, y_pred, cls_dict, clss)
    plot_cf_matrix(cf_matrix, save_path_cf_matrix, clss)
    
    sample_test_scores = convert_test_scores_to_sample_scores(test_scores, sam_dict, score_path)
    
    y_scores = sample_test_scores.bad
    y_true = sample_test_scores.sample_truth
    save_path_roc = save_path_roc.parent / 'sample_based_roc.png'
    save_path_cf_matrix = save_path_roc.parent / 'sample_based_cf_matrix.png'
    plot_negative_roc(y_true, y_scores, save_path_roc, n= 5, negative_label=cls_dict[neg])
    y_pred = create_predictions_from_threshold(y_scores, _threshold, clss, cls_dict)
    cf_matrix = calculate_cf_matrix(y_true, y_pred, clss)
    plot_cf_matrix(cf_matrix, save_path_cf_matrix, clss)


if __name__ == '__main__':
    path_to_config_json = Path(r'.\config.json')
    (path_to_scores, 
    negative_class,
    classes,
    class_dict,
    threshold,
    sample_dict) = load_config(path_to_config_json)

    main(path_to_scores, threshold, classes, negative_class, class_dict, sample_dict)
    

