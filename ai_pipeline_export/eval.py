import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
from data_utils import *
from pathlib import Path


def get_y_true_and_y_pred_for_cf_matrix(scores: pd.DataFrame, truth: pd.Series, import_params):
    def get_max_index(row):
        return row.idxmax()

    max_column = scores.apply(get_max_index, axis=1)
    y_pred = max_column.apply(lambda x: import_params.model_classes_to_index[x])
    y_true = truth.apply(lambda x: import_params.model_classes_to_index[x])
    return y_true, y_pred

def convert_scores_to_good_bad(df, import_params):
    df_good_bad = pd.DataFrame()
    df_good_bad[import_params.positive_class] = df[import_params.positive_class]
    df_good_bad['bad'] = 1 - df_good_bad
    return df_good_bad

def convert_classes_to_good_bad(ser, import_params):
    return ser.apply(lambda x: import_params.positive_class if x==import_params.positive_class else 'bad')

def get_y_binary(df, import_params):
    def convert_class_to_binary(row):
        # returns 0 or 1 depending on class and positive_index.
        # positive_index can only be 0 or 1 
        if not row == import_params.positive_class:
            return 1 - import_params.pos_label
        else:
            return import_params.pos_label

    y_binary = df.apply(convert_class_to_binary)
    return y_binary

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
    params_paths = [
        r'C:\Users\1699\Repositories\ai_results_viz\ai_pipeline_export\params_C61_collated_test.yml',
    ]
    
    # import params
    for params_path in params_paths:
        import_params = load_params(params_path, 'import_params')
        export_params = load_params(params_path, 'export_params')

        # load image based data and transformation to sample-revolution based
        # for sample-revolution based samples only the lowest good score of the revolution is considered
        df_results_image_based = load_results(import_params.import_file_path, import_params)
        df_results_sample_based = get_sample_based_scores(df_results_image_based, import_params)

        # save data frames to csv file
        export_path = Path(export_params.export_path)
        path_image_based_csv = export_path / 'image_based_scores.csv'
        path_sample_based_csv = export_path / 'sample_based_scores.csv'
        df_results_image_based.to_csv(path_image_based_csv)
        df_results_sample_based.to_csv(path_sample_based_csv)

        # Calculate image and sample based confusion matrix
        # Prediction in y_pred is the index of the class with highest score.
        scores_img = df_results_image_based[import_params.model_classes]
        truth_img = df_results_image_based.truth
        save_path_cf_matrix_img = export_path / 'image_based_multiclass_cf_matrix.png'

        scores_sample = df_results_sample_based[import_params.model_classes]
        truth_sample = df_results_sample_based.sample_truth
        save_path_cf_matrix_sample = export_path / 'sample_based_cf_matrix.png'
        
        scores_truth_path = (
            (scores_img, truth_img, save_path_cf_matrix_img),
            (scores_sample, truth_sample, save_path_cf_matrix_sample)
        )
        for scores, truth, path in scores_truth_path:
            y_true, y_pred = get_y_true_and_y_pred_for_cf_matrix(scores, truth, import_params)
            cf_matrix = calculate_cf_matrix(y_true, y_pred, import_params)
            plot_cf_matrix(cf_matrix, path, import_params.model_classes)

        # Convert image based confusion matrix to binary confusion matrix
        y_true = convert_classes_to_good_bad(truth_img, import_params)
        y_pred = scores_img.agg(lambda x: import_params.positive_class if x.idxmax() == import_params.positive_class else 'bad', axis=1)
        cf_matrix = confusion_matrix(y_true, y_pred, labels=['bad', 'good'])
        save_path_cf_matrix_img_binary = export_path / 'image_based_binary_cf_matrix.png'
        plot_cf_matrix(cf_matrix, save_path_cf_matrix_img_binary, ['bad', 'good'])

        # converting 'truth' value to binary and calculate maximum negative score for ROC curve
        # plotting and saving ROC curve
        y_good_score = df_results_image_based[import_params.positive_class]
        y_true_binary = get_y_binary(df_results_image_based.truth, import_params)
        save_path_ROC_curve = export_path / 'image_based_ROC_curve.png'
        benchmark = import_params.benchmark
        plot_ovr_frr_dr(y_true_binary, y_good_score, save_path_ROC_curve, pos_label=import_params.pos_label, benchmark=benchmark)

        # converting truth to binary and calculate maximum negative score for ROC curve
        # plotting and saving ROC curve
        y_good_score = df_results_sample_based[import_params.positive_class]
        y_true_binary = get_y_binary(df_results_sample_based.sample_truth, import_params)
        save_path_ROC_curve = export_path / 'sample_based_ROC_curve.png'
        plot_ovr_frr_dr(y_true_binary, y_good_score, save_path_ROC_curve, pos_label=import_params.pos_label, benchmark=benchmark)