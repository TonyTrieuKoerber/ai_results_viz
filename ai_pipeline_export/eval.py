import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from data_utils import *
from pathlib import Path
from typing import Tuple


def get_sample_based_scores(
        df_img: pd.DataFrame,
        import_params: ImportParams) -> pd.DataFrame:
    """Takes data frame containing model scores and converts it from
    image-based to sample/revolution-based scores. Largest 'bad' score
    and smallest 'good' score of every sample-revolution picked. Data frame
    must contain 'category', 'sample' and 'revolution' columns.

    Args:
        df_img (pd.DataFrame): Data frame containing model scores for every image
        import_params (ImportParams): Import parameters from params.yml

    Returns:
        Converted data frame containing model scores for every sample.
    """
    df_samples = df_img.groupby(
        ['category', 'sample', 'revolution']).agg(max).reset_index()
    good_min = df_img.groupby(['category', 'sample', 'revolution']).agg(
        {import_params.positive_class: 'min'}).values
    df_samples[import_params.positive_class] = good_min
    df_samples['truth'] = df_samples.category.apply(
        lambda x: import_params.category_to_model_class_map[x])
    df_samples.drop([import_params.image_name_column,
                    'trigger'], axis=1, inplace=True)
    return df_samples


def get_y_true_and_y_pred_for_multiclass_cf_matrix(
        df_scores: pd.DataFrame,
        ser_truth: pd.Series,
        import_params: ImportParams) -> Tuple[pd.Series, pd.Series]:
    """Calculates prediction 'y_pred' from model scores data frame. Class with
    maximum score is chosen as the prediction. Also, takes ground truth values as
    index values and converts them to model prediction classes (e.g. 'good',
    'bad', etc.).

    Args:
        df_scores (pd.DataFrame): Data frame containing all model prediction scores
        ser_truth (pd.Series): Series containing truth values
        import_params (ImportParams): Import parameters from params.yml

    Returns:
        Truth and prediction values for multiclass confusion matrix.
    """
    def get_max_index(row):
        return row.idxmax()

    max_column = df_scores.apply(get_max_index, axis=1)
    y_pred = max_column.apply(
        lambda x: import_params.model_classes_to_index[x])
    y_true = ser_truth.apply(lambda x: import_params.model_classes_to_index[x])
    return y_true, y_pred


def get_y_true_and_y_pred_for_binary_cf_matrix(
        df_scores: pd.DataFrame,
        ser_truth: pd.Series,
        import_params: ImportParams,
        threshold=0.5) -> Tuple[pd.Series, pd.Series]:
    """Calculates prediction 'y_pred' from data frame containing model scores.
    Prediction is 'good' if 'good'-score is above threshold. Otherwise, prediction
    will be 'bad'. Also, takes ground truth values and converts them from indexes
    to model prediction classes 'good' or 'bad'.

    Args:
        df_scores (pd.DataFrame): Data frame containing model scores
        ser_truth (pd.Series): Series of ground truth values
        import_params (ImportParams): Import parameters from params.yml
        threshold (float, optional): 'Good' prediction threshold. Defaults to 0.5.

    Returns:
        Truth and prediction values for binary confusion matrix.
    """
    y_pred = df_scores[import_params.positive_class].apply(
        lambda x: import_params.positive_class if x > threshold else 'bad')
    y_true = ser_truth.apply(lambda x: import_params.positive_class
                             if x == import_params.positive_class else 'bad')
    return y_true, y_pred


def get_threshold_from_benchmark(
        df: pd.DataFrame,
        benchmark_value: float,
        benchmark_type: str,
        import_params: ImportParams) -> float:
    """Takes data frame and benchmark (either FRR or DR). Calculates
    threshold to reach required benchmark.

    Args:
        df (pd.DataFrame): Data frame containing 'good' scores
        benchmark_value (float): FRR or DR value
        benchmark_type (str): Specify if 'FRR' or 'DR'
            should be taken from benchmark.
        import_params (ImportParams): Import parameters from
            params.yml

    Raises:
        ValueError: Benchmark type can only be 'frr' or 'dr'.

    Returns:
        float: threshold value to reach benchmark
    """
    positive_class = import_params.positive_class
    if benchmark_type.lower() == 'frr':
        good_scores = df[df['truth'] == positive_class][positive_class]
    elif benchmark_type.lower() == 'dr':
        good_scores = df[df['truth'] != positive_class][positive_class]
    else:
        raise ValueError("benchmark type has to be either 'frr' or 'dr'")

    good_scores_sorted = good_scores.sort_values()
    threshold = good_scores_sorted.iloc[
        int(len(good_scores_sorted)*benchmark_value/100)]
    return threshold


def calculate_cf_matrix(
        y_true: pd.Series,
        y_pred: pd.Series,
        class_type: str,
        import_params: ImportParams) -> np.ndarray:
    """Calculates confusion matrix from ground truth and predictions. Class_type
    ('binary' or 'multiclass') must to be specified.

    Args:
        y_true (pd.Series): Series containing ground truth values
        y_pred (pd.Series): Series containing prediction values
        class_type: Specify if predictions are 'binary' (= 'good', 'bad') or
            'multiclass' (individual class for each failed category).
        import_params (ImportParams): Import parameters from params.yml

    Returns:
        Confusion matrix ground truth vs containing prediction
    """
    if class_type == 'binary':
        cf_matrix_labels = ['bad', 'good']
    elif class_type == 'multiclass':
        cf_matrix_labels = [import_params.model_classes_to_index[x]
                            for x in import_params.model_classes]
    else:
        raise ValueError(
            '"class_type" must be either "multiclass" or "binary"')
    cf_matrix = confusion_matrix(y_true, y_pred, labels=cf_matrix_labels)
    return cf_matrix


def plot_cf_matrix_multiclass(
        cf_matrix: np.ndarray,
        path_img: str,
        classes: list,
        score_type: str = '') -> None:
    'plot and save multiclass confusion matrix'
    plt.clf()
    plt.figure()
    ax = sns.heatmap(cf_matrix, annot=True, fmt='.5g', cmap='Blues')
    ax.set_title(f'Confusion Matrix ({score_type})')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground Truth')
    ax.xaxis.set_ticklabels(classes, rotation=30)
    ax.yaxis.set_ticklabels(classes, rotation=30)
    plt.savefig(path_img, bbox_inches='tight', format='svg')


def plot_cf_matrix_binary(
        cf_matrix: np.ndarray,
        report_folder: str,
        classes: list,
        threshold: float = None,
        score_type: str = '') -> None:
    'plot and save binary confusion matrix with given threshold'
    plt.clf()
    plt.figure(figsize=[6.5, 2.4])
    ax = sns.heatmap(cf_matrix, annot=True, fmt='.5g', cmap='Blues')
    ax.set_title(f'Confusion Matrix ({score_type})')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground Truth')
    ax.xaxis.set_ticklabels(classes, rotation=30)
    ax.yaxis.set_ticklabels(classes, rotation=30)
    if not cf_matrix[0].sum() == 0:
        DR_str = f'DR = {(cf_matrix[0][0]/cf_matrix[0].sum()*100):.2f}%'
    else:
        DR_str = 'Cannot calculate DR.\nNum "bad" images or samples == 0.'
    if not cf_matrix[1].sum() == 0:
        FRR_str = f'FRR = {(cf_matrix[1][0]/cf_matrix[1].sum()*100):.2f}%'
    else:
        FRR_str = 'Cannot calculate FRR.\nNum "good" images or samples == 0.'
    textstr = f'{FRR_str}\n{DR_str}'
    if threshold:
        textstr += f'\nThreshold = {threshold:.4f}'
    plt.figtext(0.02, 0.85, textstr, fontsize='large',
                bbox=dict(facecolor='none', edgecolor='black'))
    plt.subplots_adjust(left=0.5)
    plt.savefig(report_folder, bbox_inches='tight', format='svg')


def get_df_roc(
        y_t: list,
        y_s: list,
        score_type: str,
        import_params: ImportParams,
        export_params: ExportParams) -> pd.DataFrame:
    """Takes ground truth and 'good' class-scores and calculates
    complementary ROC curve with detection rate vs false reject rate
    instead of true positive rate vs false positive rate.

    Args:
        y_t (list): ground truth data points
        y_s (list): good scores calculated by the model
        score_type (str): Specify if scores are 'image_based' or
            'sample-based'
        import_params(ImportParams): import parameters from params.yml
        export_params(ExportParams): export parameters from params.yml

    Returns:
        Data Frame containing false reject rate and detection rate
        values in percent, and the respective 'good' threshold.
    """

    export_path = export_params.export_path
    path_roc_csv = export_path / (score_type + '_complementary_ROC_curve.csv')

    fpr, tpr, thresholds = roc_curve(
        y_t, y_s, pos_label=import_params.positive_class)
    frr = 1 - tpr
    dr = 1 - fpr

    frr_percent = 100 * frr
    dr_percent = 100 * dr
    df_roc = pd.DataFrame({'frr_percent': frr_percent, 'dr_percent': dr_percent,
                           'threshold': thresholds})
    df_roc.to_csv(path_roc_csv, index=False)

    return df_roc


def plot_complementary_roc(
        df_roc: pd.DataFrame, score_type: str, export_params: ExportParams,
        n=None, benchmark=None,) -> None:
    """Takes data frame containing false reject rate, detection rate
    and threshold and creates complementary ROC plot.
    Only every nth threshold is shown in plot.

    Args:
        df_roc (pd.DataFrame): Data frame containing false reject rate,
            detection rate and threshold values.
        score_type (str): Specify if scores are 'image_based' or
            'sample-based'
        export_params(ExportParams): export parameters from params.yml
        n (int): Plot every nth threshold. Will be set to 1 if n==0.
        benchmark (list): plot classical vision detection benchmark in ROC
    """

    def reduce_to_n_values(df, col, n):
        return np.append(df[col].iloc[::n], df[col].iloc[-1])

    export_path = export_params.export_path
    path_roc_img = export_path / (score_type + '_complementary_ROC_curve.' +
                                  export_params.img_format)
    path_roc_img_no_statistics = export_path / (score_type +
                                                '_complementary_ROC_curve(no_statistics).' + export_params.img_format)

    df_roc_x_lim = df_roc[df_roc.frr_percent < export_params.x_lim]
    auc_x_lim = auc(df_roc_x_lim.frr_percent/100, df_roc_x_lim.dr_percent/100)
    auc_total = auc(df_roc.frr_percent/100, df_roc.dr_percent/100)

    if not n:
        n = int(len(df_roc_x_lim)/20)
    if n == 0:
        n = 1
    frr_percent = reduce_to_n_values(df_roc_x_lim, 'frr_percent', n)
    dr_percent = reduce_to_n_values(df_roc_x_lim, 'dr_percent', n)
    thresholds = reduce_to_n_values(df_roc_x_lim, 'threshold', n)

    plt.clf()
    plt.figure()
    plt.plot(frr_percent, dr_percent, 'o-',
             color='darkorange', lw=1, label='threshold')
    plt.plot([0, 100], [0, 100], color='navy', lw=1, linestyle='--')
    if benchmark:
        plt.plot(benchmark['frr'], benchmark['dr'], 'gs',
                 label='vision detection')
    plt.xlim([0.0, export_params.x_lim])
    plt.ylim([0.0, 105])
    plt.xlabel('False Reject Rate [%]')
    plt.ylabel('Detection Rate [%]')
    plt.title(
        f'Complementary receiver operating characteristic ({score_type})', loc='right')
    plt.legend(loc='lower right')
    plt.savefig(path_roc_img_no_statistics,
                format=export_params.img_format, transparent=True)

    auc_text = f'Area under curve (AUC)\nTotal = {auc_total:.3f}'
    auc_text += f'\nx<{export_params.x_lim}% = {auc_x_lim:.3f}'
    plt.figtext(0.63, 0.25, auc_text, fontsize='medium')
    for x, y, txt in zip(frr_percent, dr_percent, thresholds):
        plt.annotate(np.round(txt, 3), (x, y-0.04),
                     rotation=-30, verticalalignment='top')
    plt.savefig(path_roc_img,
                format=export_params.img_format, transparent=True)


def plot_summary_roc(
        export_params: ExportParams,
        benchmark: dict = None,
        show_thresholds: bool = False):
    """Placeholder"""
    path_roc_csv_img = export_params.export_path /\
        'image_based_complementary_ROC_curve.csv'
    path_roc_csv_sample = export_params.export_path /\
        'sample_based_complementary_ROC_curve.csv'
    df_img = pd.read_csv(path_roc_csv_img)
    df_sample = pd.read_csv(path_roc_csv_sample)
    plt.clf()
    plt.figure()
    plt.plot(df_img.frr_percent, df_img.dr_percent, '.-',
             color='darkorange', lw=1, label='image-based thresholds')
    plt.plot(df_sample.frr_percent, df_sample.dr_percent, '.-',
             color='blue', lw=1, label='sample-based thresholds')
    plt.plot([0, 100], [0, 100], color='navy', lw=1, linestyle='--')
    if benchmark:
        plt.plot(benchmark['frr'], benchmark['dr'], 'gs',
                 label='vision detection')
    plt.xlim([0.0, export_params.x_lim])
    plt.ylim([0.0, 105])
    plt.xlabel('False Reject Rate [%]')
    plt.ylabel('Detection Rate [%]')
    plt.title(
        'Complementary receiver operating characteristic', loc='right')
    plt.legend(loc='lower right')
    plt.savefig(
        export_params.export_path /
        ('Complementary ROC curve.'+export_params.img_format),
        format=export_params.img_format, transparent=True)


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
        df_results_image_based = load_results(
            import_params.import_file_path, import_params)
        df_results_sample_based = get_sample_based_scores(
            df_results_image_based, import_params)

        # save data frames to csv file
        export_path = Path(export_params.export_path)
        path_image_based_csv = export_path / 'image_based_scores.csv'
        path_sample_based_csv = export_path / 'sample_based_scores.csv'
        df_results_image_based.to_csv(path_image_based_csv, index=False)
        df_results_sample_based.to_csv(path_sample_based_csv, index=False)

        # Arranging data for image and sample based evaluation
        # Storing data in eval_params
        eval_params = {
            'image_based':
            {
                'data_frame': df_results_image_based,
                'scores': df_results_image_based[import_params.model_classes],
                'truth': df_results_image_based.truth
            },
            'sample_based':
            {
                'data_frame': df_results_sample_based,
                'scores': df_results_sample_based[import_params.model_classes],
                'truth': df_results_sample_based.truth
            }
        }

        for score_type, data in eval_params.items():
            scores = data['scores']
            truth = data['truth']

            # create multiclass confusion matrix
            # prediction type: one vs one
            y_true_img, y_pred_img = get_y_true_and_y_pred_for_multiclass_cf_matrix(
                scores, truth, import_params)
            cf_matrix_img = calculate_cf_matrix(
                y_true_img, y_pred_img, 'multiclass', import_params)
            img_path_mcfm = export_path / \
                (score_type + '_multiclass_cf_matrix.svg')
            plot_cf_matrix_multiclass(
                cf_matrix_img, img_path_mcfm, import_params.model_classes, score_type=score_type)

            # create ROC curve
            # prediction type: good score > threshold
            y_good_score = scores[import_params.positive_class]
            df_roc = get_df_roc(truth, y_good_score, score_type, import_params,
                                export_params)
            plot_complementary_roc(df_roc, score_type, export_params,
                                benchmark=import_params.benchmark)

            # determine thresholds to reach benchmark frr and dr
            # skip if number of good samples == 0
            if (score_type == 'sample_based' and
                    not import_params.positive_class in truth.unique()):
                continue
            frr_dr_threshold = {}

            benchmark_1 = import_params.benchmark['frr']
            benchmark_2 = 0.2 * benchmark_1
            benchmark_3 = import_params.benchmark['dr']

            frr_dr_threshold['FRR_benchmark'] = \
                get_threshold_from_benchmark(
                    data['data_frame'], benchmark_1, 'frr', import_params)
            frr_dr_threshold['FRR_benchmark_20_percent'] = \
                get_threshold_from_benchmark(
                    data['data_frame'], benchmark_2, 'frr', import_params)
            frr_dr_threshold['DR_benchmark'] = \
                get_threshold_from_benchmark(
                    data['data_frame'], benchmark_3, 'dr', import_params)

            for key, threshold in frr_dr_threshold.items():
                # create binary confusion matrix
                # prediction type: good score > threshold
                y_true_image_binary, y_pred_img_binary = get_y_true_and_y_pred_for_binary_cf_matrix(
                    scores, truth, import_params, threshold=threshold)
                cf_matrix_sample = calculate_cf_matrix(
                    y_true_image_binary, y_pred_img_binary, 'binary', import_params)
                img_dir_bcfm = export_path / f'{key}'
                img_dir_bcfm.mkdir(exist_ok=True)
                img_path_bcfm = img_dir_bcfm / \
                    (score_type + '_binary_cf_matrix.svg')
                plot_cf_matrix_binary(cf_matrix_sample, img_path_bcfm, ['bad', 'good'],
                                    threshold=threshold, score_type=score_type)

        path_roc_csv_img = export_params.export_path /\
            'image_based_complementary_ROC_curve.csv'
        path_roc_csv_sample = export_params.export_path /\
            'sample_based_complementary_ROC_curve.csv'
        if (path_roc_csv_img.is_file() and
                path_roc_csv_sample.is_file()):
            plot_summary_roc(export_params, benchmark=import_params.benchmark)
