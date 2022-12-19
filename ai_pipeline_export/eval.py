import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from data_utils import *
from pathlib import Path


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
    y_pred = scores[import_params.positive_class].apply(lambda x: import_params.positive_class if x > threshold else 'bad')
    y_true = truth.apply(lambda x: import_params.positive_class if x==import_params.positive_class else 'bad')
    return y_true, y_pred

def get_truth_type(score_type):
    """Placeholder"""
    truth_type = 'image_truth' if score_type == 'image_based' else 'sample_truth'
    return truth_type

def get_dr_and_threshold(df, frr, score_type, import_params):
    truth_type = get_truth_type(score_type)    
    sorted_good_scores_good = df[df[truth_type] == import_params.positive_class][import_params.positive_class].sort_values()
    threshold = sorted_good_scores_good.iloc[int(len(sorted_good_scores_good)*frr)]
    good_scores_bad = df[df[truth_type] != import_params.positive_class][import_params.positive_class]
    dr = good_scores_bad.le(threshold).mean()
    return dr, threshold

def get_frr_and_threshold(df, dr, score_type, import_params):
    truth_type = get_truth_type(score_type)
    sorted_good_scores_bad = df[df[truth_type] != import_params.positive_class][import_params.positive_class].sort_values()
    threshold = sorted_good_scores_bad.iloc[int(len(sorted_good_scores_bad)*dr)]
    good_scores_good = df[df[truth_type] == import_params.positive_class][import_params.positive_class]
    frr = good_scores_good.le(threshold).mean()
    return frr, threshold

def calculate_cf_matrix(y_true:pd.Series, y_pred:pd.Series, import_params) -> np.ndarray:
    cf_matrix_labels = [import_params.model_classes_to_index[x] for x in import_params.model_classes]
    cf_matrix = confusion_matrix(y_true, y_pred, labels=cf_matrix_labels)
    return cf_matrix

def plot_cf_matrix_multiclass(cf_matrix: np.ndarray, report_folder:str, classes:list, 
    threshold:float = None, score_type:str = ""):
    "plot and save confusion matrix"
    plt.clf()
    plt.figure()
    ax = sns.heatmap(cf_matrix, annot=True, fmt='.5g', cmap='Blues')
    ax.set_title(f'Confusion Matrix ({score_type})')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground Truth')
    ax.xaxis.set_ticklabels(classes, rotation=30)
    ax.yaxis.set_ticklabels(classes, rotation=30)
    plt.savefig(report_folder,bbox_inches = "tight")

def plot_cf_matrix_binary(cf_matrix: np.ndarray, report_folder:str, classes:list, 
    threshold:float = None, score_type:str = ""):
    "plot and save confusion matrix"
    plt.clf()
    plt.figure(figsize=[8, 4.8])
    ax = sns.heatmap(cf_matrix, annot=True, fmt='.5g', cmap='Blues')
    ax.set_title(f'Confusion Matrix ({score_type})')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground Truth')
    ax.xaxis.set_ticklabels(classes, rotation=30)
    ax.yaxis.set_ticklabels(classes, rotation=30)
    if not (sum(cf_matrix[0]) == 0 or sum(cf_matrix[1]) == 0):
        frr = cf_matrix[1][0]/cf_matrix[1].sum()
        dr = cf_matrix[0][0]/cf_matrix[0].sum()
        textstr = f'FRR = {(frr*100):.2f}%\nDR = {(dr*100):.2f}%'
        if threshold:
            textstr += f'\nThreshold = {threshold:.4f}'
        plt.figtext(0.02, 0.85, textstr, fontsize='large', bbox=dict(facecolor='none', edgecolor='black'))
        plt.subplots_adjust(left=0.25)
    plt.savefig(report_folder,bbox_inches = "tight")

def plot_ovr_frr_dr(
    y_t: list, 
    y_s: list, 
    save_path: str, 
    n = None, 
    xlim = 15.0, 
    pos_label=1, 
    benchmark = None,
    score_type:str =""):
    """Takes ground truth and scores for good class and plots complementary ROC curve with
    detection rate vs false reject rate instead of true positive rate vs false
    positive rate. Only every nth threshold is shown in plot.
    Only for binary classification.

    Args:
        y_t (list): ground truth data points
        y_s (list): scores calculated by the model
        save_path (str): path to save the roc curve
        n (int): only every nth threshold is plotted
        xlim (float): maximum value of x axis in %
        benchmark (list): plot classical vision detection benchmark in ROC
        negative_label (int, optional): negative label of ground truth. Defaults to 1.
    """
    plt.clf()
    fpr, tpr, thresholds = roc_curve(y_t, y_s, pos_label=pos_label)
    frr = 100 * (1 - tpr)
    dr = 100 * (1 - fpr)
    
    roc_points = list(zip(frr, dr))
    if xlim:
        frr_auc_xlim = [x[0] for x in roc_points if x[0] < xlim]
        dr_auc_xlim = [x[1] for x in roc_points if x[0] < xlim]
        auc_xlim = auc(frr_auc_xlim, dr_auc_xlim)
    auc_max = auc(frr, dr)
    
    if not n:
        n = int(len(fpr)/100)
    elif n == 0:
        n = 1
    frr = np.append(frr[::n], frr[-1])
    dr = np.append(dr[::n], dr[-1])
    thresholds = np.append(thresholds[::n], thresholds[-1])

    plt.figure()
    plt.plot(frr, dr, 'o-', color='darkorange', lw=1, label='threshold')
    plt.plot([0, 100], [0, 100], color='navy', lw=1, linestyle='--')
    if benchmark:
        plt.plot(*benchmark, 'go', label='vision detection')
    plt.xlim([0.0, xlim])
    plt.ylim([0.0, 105])
    plt.xlabel('False Reject Rate [%]')
    plt.ylabel('Detection Rate [%]')
    plt.title(f'Complementary receiver operating characteristic ({score_type})', loc='right')
    auc_text = f'Area under curve (AUC)\nTotal = {auc_max:.3f}'
    if xlim:
        auc_text += f'\n0<x<{xlim}% = {auc_xlim:.3f}'
    plt.figtext(0.63, 0.25, auc_text, fontsize='medium')
    plt.legend(loc="lower right")
    for x, y, txt in zip(frr,dr,thresholds):
        plt.annotate(np.round(txt,3), (x, y-0.04), rotation = -30, verticalalignment ='top')
    plt.savefig(save_path)


if __name__ == '__main__':
    params_paths = [
        r'C:\Users\1699\Repositories\ai_results_viz\ai_pipeline_export\params_C61_collated test.yml',
    ]
    
    # import params
    for params_path in params_paths:
        import_params = load_params(params_path, 'import_params')
        export_params = load_params(params_path, 'export_params')

        # load image based data and transformation to sample-revolution based
        # for sample-revolution based data only the lowest good score of the revolution is considered
        df_results_image_based = load_results(import_params.import_file_path, import_params)
        df_results_sample_based = get_sample_based_scores(df_results_image_based, import_params)

        # save data frames to csv file
        export_path = Path(export_params.export_path)
        path_image_based_csv = export_path / 'image_based_scores.csv'
        path_sample_based_csv = export_path / 'sample_based_scores.csv'
        df_results_image_based.to_csv(path_image_based_csv)
        df_results_sample_based.to_csv(path_sample_based_csv)
        
        # Arranging data for image and sample based evaluation
        # Storing data in eval_params
        scores_img = df_results_image_based[import_params.model_classes]
        truth_img = df_results_image_based.image_truth
        scores_sample = df_results_sample_based[import_params.model_classes]
        truth_sample = df_results_sample_based.sample_truth
        
        eval_params = {
            'image_based':
            {
                'data_frame': df_results_image_based,
                'scores': scores_img,
                'truth': truth_img
            },
            'sample_based':
            {
                'data_frame': df_results_sample_based,
                'scores': scores_sample,
                'truth': truth_sample
            }
        }

        for score_type in list(eval_params.keys()):
            scores = eval_params[score_type]['scores']
            truth = eval_params[score_type]['truth']

            # create multiclass confusion matrix
            # prediction type: one vs one
            y_true_img, y_pred_img = get_y_true_and_y_pred_for_multiclass_cf_matrix(scores, truth, import_params)
            cf_matrix_img = calculate_cf_matrix(y_true_img, y_pred_img, import_params)
            img_path_mcfm = export_path / (score_type + '_multiclass_cf_matrix.png')
            plot_cf_matrix_multiclass(cf_matrix_img, img_path_mcfm, import_params.model_classes, score_type=score_type)

            # create ROC curve
            # prediction type: good score > threshold
            y_good_score = scores[import_params.positive_class]
            img_path_roc = export_path / (score_type + '_complementary_ROC_curve.png')
            plot_ovr_frr_dr(truth, y_good_score, img_path_roc, pos_label=import_params.positive_class, 
                            benchmark=import_params.benchmark, score_type=score_type)
            
            # determine thresholds to reach benchmark frr and dr
            # skip if number of good samples == 0
            if score_type == "sample_based" and sum(truth_sample==import_params.positive_class)==0:
                continue
            frr_dr_threshold = []
            frr_benchmark = import_params.benchmark['frr'] * 0.01
            frr_benchmark_80_percent = 0.8 * frr_benchmark
            dr_resulting, threshold_resulting = get_dr_and_threshold(
                eval_params[score_type]['data_frame'], frr_benchmark, score_type, import_params)
            frr_dr_threshold.append([frr_benchmark,dr_resulting,threshold_resulting])
            dr_benchmark = import_params.benchmark['dr'] * 0.01
            frr_resulting, threshold_resulting =get_frr_and_threshold(
                eval_params[score_type]['data_frame'], dr_benchmark, score_type, import_params)
            frr_dr_threshold.append([frr_resulting,dr_benchmark,threshold_resulting])
            
            for frr, dr, threshold in frr_dr_threshold:
                # create binary confusion matrix
                # prediction type: good score > threshold
                y_true_image_binary, y_pred_img_binary = get_y_true_and_y_pred_for_binary_cf_matrix(
                    scores, truth, import_params, threshold=threshold)
                cf_matrix_sample = confusion_matrix(y_true_image_binary, y_pred_img_binary, labels=['bad', 'good'])
                img_dir_bcfm = export_path / f'{score_type}_FRR_{frr:.5f}__DR_{dr:.5f}__threshold_{threshold:.5f}'
                img_dir_bcfm.mkdir(exist_ok=True)
                img_path_bcfm = img_dir_bcfm / (score_type + '_binary_cf_matrix.png')
                plot_cf_matrix_binary(cf_matrix_sample, img_path_bcfm, ['bad', 'good'], 
                    threshold=threshold, score_type=score_type)
            
