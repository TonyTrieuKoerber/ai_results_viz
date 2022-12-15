import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
from data_utils import *
from pathlib import Path
from decimal import Decimal


def get_sample_based_scores(df: pd.DataFrame, import_params):
    df_samples =  df.groupby(['category','sample','revolution']).agg(max).reset_index()
    good_min = df.groupby(['category','sample','revolution']).agg({import_params.positive_class: 'min'}).values
    df_samples['good'] = good_min
    df_samples['sample_truth'] = df_samples.category.apply(lambda x: import_params.category_to_model_class_map[x])
    return df_samples

def get_y_true_and_y_pred_for_multiclass_cf_matrix(scores: pd.DataFrame, truth: pd.Series, import_params):
    def get_max_index(row):
        return row.idxmax()

    max_column = scores.apply(get_max_index, axis=1)
    y_pred = max_column.apply(lambda x: import_params.model_classes_to_index[x])
    y_true = truth.apply(lambda x: import_params.model_classes_to_index[x])
    return y_true, y_pred

def get_y_true_and_y_pred_for_binary_cf_matrix(scores: pd.DataFrame, truth: pd.Series, import_params, threshold=0.5):
    # y_pred = scores.agg(lambda x: import_params.positive_class if x.idxmax() == import_params.positive_class else 'bad', axis=1)
    y_pred = scores[import_params.positive_class].apply(lambda x: import_params.positive_class if x > threshold else 'bad')
    y_true = truth.apply(lambda x: import_params.positive_class if x==import_params.positive_class else 'bad')
    return y_true, y_pred

def calculate_cf_matrix(y_true:pd.Series, y_pred:pd.Series, import_params) -> np.ndarray:
    cf_matrix_labels = [import_params.model_classes_to_index[x] for x in import_params.model_classes]
    cf_matrix = confusion_matrix(y_true, y_pred, labels=cf_matrix_labels)
    return cf_matrix

def plot_cf_matrix(cf_matrix: np.ndarray, report_folder:str, classes:list):
    "plot and save confusion matrix"
    plt.clf()
    ax = sns.heatmap(cf_matrix, annot=True, fmt='.5g', cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground Truth')
    ax.xaxis.set_ticklabels(classes, rotation=30)
    ax.yaxis.set_ticklabels(classes, rotation=30)
    plt.savefig(report_folder,bbox_inches = "tight")

def plot_ovr_frr_dr(y_t: list, y_s: list, save_path: str, n = None, pos_label=1, benchmark = [0,0]):
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
    plt.clf()
    fpr, tpr, thresholds = roc_curve(y_t, y_s, pos_label=pos_label)
    if not n:
        n = int(len(fpr)/20)

    frr = 100*(1 - tpr[::n])
    dr = 100*(1 - fpr[::n])
    thresholds = thresholds[::n]

    plt.figure()
    plt.plot(frr, dr, 'o-', color='darkorange', lw=1, label='threshold')
    plt.plot([0, 100], [0, 100], color='navy', lw=1, linestyle='--')
    # plt.plot(*benchmark, 'go', label='vision detection')
    plt.xlim([0.0, 100])
    plt.ylim([0.0, 105])
    plt.xlabel('False Reject Rate [%]')
    plt.ylabel('Detection Rate [%]')
    plt.title('Complementary receiver operating characteristic (image-based evaluation)')
    plt.legend(loc="lower right")
    for x, y, txt in zip(frr,dr,thresholds):
        plt.annotate(np.round(txt,3), (x, y-0.04), rotation = -30, verticalalignment ='top')
    plt.savefig(save_path)

def get_dr_and_threshold(df, frr, truth_type = 'image_truth'):
    sorted_good_scores_good = df[df[truth_type] == import_params.positive_class][import_params.positive_class].sort_values()
    threshold = sorted_good_scores_good.iloc[int(len(sorted_good_scores_good)*frr)]
    good_scores_bad = df[df[truth_type] != import_params.positive_class][import_params.positive_class]
    dr = good_scores_bad.le(threshold).mean()
    return dr, threshold

def get_frr_and_threshold(df, dr, truth_type = 'image_truth'):
    sorted_good_scores_bad = df[df[truth_type] != import_params.positive_class][import_params.positive_class].sort_values()
    threshold = sorted_good_scores_bad.iloc[int(len(sorted_good_scores_bad)*dr)]
    good_scores_good = df[df[truth_type] == import_params.positive_class][import_params.positive_class]
    frr = good_scores_good.le(threshold).mean()
    return frr, threshold


if __name__ == '__main__':
    params_paths = [
        r'C:\Users\1699\Repositories\ai_results_viz\ai_pipeline_export\params.yml',
    ]
    
    # import params
    for params_path in params_paths:
        import_params = load_params(params_path, 'import_params')
        export_params = load_params(params_path, 'export_params')

        # load image based data and transformation to sample-revolution based
        # for sample-revolution based data only the lowest good score of the revolution is considered
        df_results_image_based = pd.read_csv(import_params.import_file_path)
        # df_results_sample_based = get_sample_based_scores(df_results_image_based, import_params)

        # save data frames to csv file
        export_path = Path(export_params.export_path)

        # Calculate image and sample based confusion matrices
        # Prediction in y_pred is the index of the class with highest score.
        scores_img = df_results_image_based[import_params.model_classes]
        truth_img = df_results_image_based.image_truth

        frr_dr_threshold = []
        frr_benchmark = import_params.benchmark[1] * 0.01
        dr_resulting, threshold_resulting = get_dr_and_threshold(df_results_image_based, frr_benchmark)
        frr_dr_threshold.append([frr_benchmark,dr_resulting,threshold_resulting])
        dr_benchmark = import_params.benchmark[0] * 0.01
        frr_resulting, threshold_resulting =get_frr_and_threshold(df_results_image_based, dr_benchmark)
        frr_dr_threshold.append([frr_resulting,dr_benchmark,threshold_resulting])

        # scores_truth_path = (
        #     (scores_img, truth_img, save_paths_img_based_cf_matrices),
        #     # (scores_sample, truth_sample, save_paths_sample_based_cf_matrices)
        # )
        # for scores, truth, paths in scores_truth_path:
        path_roc = export_path / 'image_based_ROC_curve.png'
        y_good_score = scores_img[import_params.positive_class]
        plot_ovr_frr_dr(truth_img, y_good_score, path_roc, pos_label=import_params.positive_class, 
                            benchmark=import_params.benchmark)

        for frr, dr, threshold in frr_dr_threshold:
            export_folder = export_path / f'FRR_{frr:.5f}__DR_{dr:.5f}__threshold_{threshold:.5f}'
            export_folder.mkdir(exist_ok=True)
            path_cf_matrix = export_folder / 'image_based_binary_cf_matrix.png'
            y_true_binary, y_pred_binary = get_y_true_and_y_pred_for_binary_cf_matrix(scores_img, truth_img, import_params, threshold=threshold)
            # cf_matrix_img = calculate_cf_matrix(y_true_img, y_pred_img, import_params)
            cf_matrix_binary = confusion_matrix(y_true_binary, y_pred_binary, labels=['bad', 'good'])
            # plot_cf_matrix(cf_matrix_img, paths[0], import_params.model_classes)
            plot_cf_matrix(cf_matrix_binary, path_cf_matrix, ['bad', 'good'])
