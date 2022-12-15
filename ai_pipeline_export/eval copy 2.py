import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
from data_utils import *
from pathlib import Path
from decimal import Decimal


def get_sample_based_scores(df: pd.DataFrame, import_params):
    """Placeholder"""
    df_samples = df.groupby(
        ['category', 'sample', 'revolution']).agg(max).reset_index()
    good_min = df.groupby(['category', 'sample', 'revolution']).agg(
        {import_params.positive_class: 'min'}).values
    df_samples['good'] = good_min
    df_samples['sample_truth'] = df_samples.category.apply(
        lambda x: import_params.category_to_model_class_map[x])
    return df_samples


def get_y_true_and_y_pred_for_multiclass_cf_matrix(scores: pd.DataFrame, truth: pd.Series, import_params):
    """Placeholder"""
    def get_max_index(row):
        return row.idxmax()

    max_column = scores.apply(get_max_index, axis=1)
    y_pred = max_column.apply(
        lambda x: import_params.model_classes_to_index[x])
    y_true = truth.apply(lambda x: import_params.model_classes_to_index[x])
    return y_true, y_pred


def get_y_true_and_y_pred_for_binary_cf_matrix(scores: pd.DataFrame, truth: pd.Series, import_params):
    """Placeholder"""
    y_pred = scores.agg(lambda x: import_params.positive_class if x.idxmax(
    ) == import_params.positive_class else 'bad', axis=1)
    y_true = truth.apply(lambda x: import_params.positive_class if x ==
                         import_params.positive_class else 'bad')
    return y_true, y_pred


def calculate_cf_matrix(y_true: pd.Series, y_pred: pd.Series, import_params) -> np.ndarray:
    """Placeholder"""
    cf_matrix_labels = [import_params.model_classes_to_index[x]
                        for x in import_params.model_classes]
    cf_matrix = confusion_matrix(y_true, y_pred, labels=cf_matrix_labels)
    return cf_matrix


def plot_cf_matrix(cf_matrix: np.ndarray, report_folder: str, classes: list):
    """plot and save confusion matrix"""
    plt.clf()
    ax = sns.heatmap(cf_matrix, annot=True, fmt='.5g', cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground Truth')
    ax.xaxis.set_ticklabels(classes, rotation=30)
    ax.yaxis.set_ticklabels(classes, rotation=30)
    plt.savefig(report_folder, bbox_inches='tight')


def plot_ovr_frr_dr(y_t: list, y_s: list, save_path: str, n=None, pos_label=1, benchmark=None):
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
    benchmark = [0, 0] if benchmark is None else benchmark
    plt.clf()
    fpr, tpr, thresholds = roc_curve(y_t, y_s, pos_label=pos_label)
    if not n:
        n = int(len(fpr)/10) if int(len(fpr)/10) != 0 else 1
    logging.info(f'{int(len(fpr)/20)}, {len(tpr)}, {len(tpr[::n])} ')
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
    plt.legend(loc='lower right')
    for x, y, txt in zip(frr, dr, thresholds):
        plt.annotate(np.round(txt, 3), (x, y-0.04),
                     rotation=-30, verticalalignment='top')
    plt.savefig(save_path)


def evaluation(
    import_file_path: Path,
    export_path: Path,
    category_to_model_class_map: dict,
    benchmark: list = None,
    positive_class: str = 'good',
    extract_info_from_file_name: bool = True,
    regex_expression_file_name: str = '^(.+?)_[(good)|(bad)].+_(\d+)_(\d+)_(\d+)'
):
    """Placeholder"""

    import_params = ImportParams(
        import_file_path=import_file_path,
        category_to_model_class_map=category_to_model_class_map,
        positive_class=positive_class,
        extract_info_from_file_name=extract_info_from_file_name,
        regex_expression_file_name=regex_expression_file_name,
        benchmark=benchmark
    )
    export_params = ExportParams(
        export_path=export_path
    )

    # load image based data and transformation to sample-revolution based
    # for sample-revolution based samples only the lowest good score of the revolution is considered
    df_results_image_based = load_results(
        import_params.import_file_path, import_params)
    df_results_sample_based = get_sample_based_scores(
        df_results_image_based, import_params)

    # save data frames to csv file
    export_path = Path(export_params.export_path)
    path_image_based_csv = export_path / 'image_based_scores.csv'
    path_sample_based_csv = export_path / 'sample_based_scores.csv'
    df_results_image_based.to_csv(path_image_based_csv)
    df_results_sample_based.to_csv(path_sample_based_csv)

    # Calculate image and sample based confusion matrices
    # Prediction in y_pred is the index of the class with highest score.
    scores_img = df_results_image_based[import_params.model_classes]
    truth_img = df_results_image_based.image_truth
    save_paths_img_based_cf_matrices = (
        export_path / 'image_based_multiclass_cf_matrix.png',
        export_path / 'image_based_binary_cf_matrix.png',
        export_path / 'image_based_ROC_curve.png'
    )

    scores_sample = df_results_sample_based[import_params.model_classes]
    truth_sample = df_results_sample_based.sample_truth
    save_paths_sample_based_cf_matrices = (
        export_path / 'sample_based_cf_matrix.png',
        export_path / 'sample_based_binary_cf_matrix.png',
        export_path / 'sample_based_ROC_curve.png'
    )

    scores_truth_path = (
        (scores_img, truth_img, save_paths_img_based_cf_matrices),
        (scores_sample, truth_sample, save_paths_sample_based_cf_matrices)
    )
    for scores, truth, paths in scores_truth_path:
        y_true_img, y_pred_img = get_y_true_and_y_pred_for_multiclass_cf_matrix(
            scores, truth, import_params)
        y_true_sample, y_pred_sample = get_y_true_and_y_pred_for_binary_cf_matrix(
            scores, truth, import_params)
        y_good_score = scores[import_params.positive_class]
        cf_matrix_img = calculate_cf_matrix(
            y_true_img, y_pred_img, import_params)
        cf_matrix_sample = confusion_matrix(
            y_true_sample, y_pred_sample, labels=['bad', 'good'])
        plot_cf_matrix(cf_matrix_img, paths[0], import_params.model_classes)
        plot_cf_matrix(cf_matrix_sample, paths[1], ['bad', 'good'])
        plot_ovr_frr_dr(truth, y_good_score, paths[2], pos_label=import_params.positive_class,
                        benchmark=import_params.benchmark)


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
