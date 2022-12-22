import pandas as pd
import re
from typing import Union
from pathlib import Path
import yaml, json


class ImportParams:
    """Class for importing import_params field of params.yml

    Attributes:
        import_file_path (Path): Path to test_scores.csv file.
        classes (list): Model classes. Will be automatically created from classes.json if not
            specified. Defaults to None.
        model_classes_to_index (dict): Maps class to prediction index. Will be automatically
            created from classes.json if not specified. Defaults to None.
        category_to_model_class_map (dict): In case of collated classes, original classes need
            to be mapped to collated classes. If not collated, classes just map to themselves.
        positive_class (str): Name of 'good' class.
        extract_info_from_file_name: If set to True, 'category', 'sample', 'revolution' and
            'trigger' information will be extracted from file name
        regex_expression_file_name: regex expr to extact category information from file name.
        benchmark (dict): Benchmark (false reject rate and detection rate) of classic vision
            detection for this camera station. Used for reference in ROC curve and CF matrix.

    Example:
        >>> params_path = Path('./config/params.yml')
        >>> with open(params_path, 'r') as file:\
        >>>    params = yaml.safe_load(file)
        >>> import_params = ImportParams(**params['import_params'])

    """

    import_file_path: Path
    model_classes: list = None
    model_classes_to_index: dict = None
    category_to_model_class_map: dict
    positive_class: str
    extract_info_from_file_name: bool = True
    regex_expression_file_name: str = '^(.+?)_[(good)|(bad)].+_(\d+)_(\d+)_(\d+)'
    benchmark: dict

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.import_file_path = Path(self.import_file_path)
        if self.model_classes is None:
            class_json_path = self.import_file_path.parent / 'final' / 'classes.json'
            with open(class_json_path) as f:
                self.model_classes = json.load(f)
            self.model_classes_to_index = {
                j: i for i, j in enumerate(self.model_classes)}
        self.image_name_column = 'img'
        self.score_column = 'scores'
        self.truth_column = 'truth'


class ExportParams:
    """Class for importing export_params field of params.yml

    Attributes:
        export_path (Path): Path to saves image and csv-files.
        img_format (str): Format in which images are saved
        x_lim (float): Maximum x axis value of ROC plot
    """
    export_path: Path
    img_format: str = 'svg'
    x_lim: float = 15

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.export_path = Path(self.export_path)


def load_params(
    path: str,
    params_type: str) \
        -> Union[ImportParams, ExportParams]:
    """Loads parameters from yml file. Parameter type ('ImportParams' or
    'ExportParams') need to be specified.
    """
    params_path = Path(path)
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    if params_type == 'import_params':
        return ImportParams(**params[params_type])
    return ExportParams(**params[params_type])


def extract_info(
        df_in: pd.DataFrame,
        import_params: ImportParams,
        extract_info_from_file_name: bool = False) -> pd.DataFrame:
    """Takes data frame containing the original data and splits scores column into
    individual score columns. Converts ground truth column from index to class name.
    If desired, extracts 'category', 'sample', 'revolution' and 'trigger' information
    from file name via regular expression.

    Args:
        df_in (pd.DataFrame): Original data frame imported from test_scores.csv
        import_params: (ImportParams): Import parameters from params.yml
        extract_info_from_file_name (bool): Set to True to extract 'category',
            'sample', 'revolution' and 'trigger' information from file name.

    Returns:
        Data frame with extracted information.
    """

    def get_score_list_from_string(scores_raw: str):
        return list(map(float, scores_raw[1:-1].split()))

    def get_ground_truth(file_name):
        return Path(file_name).parent.name

    def get_file_infos(file_path):
        p = Path(file_path)
        file_name = p.name
        return list(re.search(import_params.regex_expression_file_name, file_name).groups())

    df_columns = [
        import_params.image_name_column,
        'category',
        'sample',
        'revolution',
        'trigger',
        import_params.truth_column
    ]

    df_out = pd.DataFrame(columns=df_columns)
    df_out[import_params.image_name_column] = df_in[import_params.image_name_column]
    # convert original scores from string to indiviual columns
    scores = df_in[import_params.score_column].apply(
        get_score_list_from_string)
    for n, category in enumerate(import_params.model_classes):
        df_out[category] = scores.apply(lambda x: x[n])
    # convert model prediction from index to classs name
    df_out[import_params.truth_column] = \
        df_out[import_params.image_name_column].apply(get_ground_truth)
    # extract infos from file name
    if extract_info_from_file_name:
        img_infos = df_out[import_params.image_name_column].apply(
            get_file_infos)
        df_out['category'] = img_infos.apply(lambda x: x[0])
        df_out['sample'] = img_infos.apply(lambda x: int(x[1]))
        df_out['revolution'] = img_infos.apply(lambda x: int(x[2]))
        df_out['trigger'] = img_infos.apply(lambda x: int(x[3]))
    return df_out


def load_results(
        path_scores: Path,
        import_params: ImportParams) -> pd.DataFrame:
    """Loads prediction results from test_scores.csv

    Args:
        path_scores (Path): Path to scores
        import_params (ImportParams): Import parameters from params.yml

    Returns:
        Data frame containing image path, model scores and groung truth.
    """

    df_results = pd.read_csv(path_scores,
                             names=[
                                 import_params.image_name_column,
                                 import_params.score_column,
                                 import_params.truth_column]
                             )
    df_results = extract_info(df_results, import_params,
                              extract_info_from_file_name=import_params.extract_info_from_file_name)
    return df_results

if __name__ == '__main__':
    params_path = r'C:\Users\1699\Repositories\ai_results_viz\ai_pipeline_export\params_C51_W1.yml'
    import_params = load_params(params_path, 'import_params')
    df_results = load_results(import_params.import_file_path, import_params)
