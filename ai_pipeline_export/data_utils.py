import pandas as pd
import re
from typing import List
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
            to be mapped to collated classes. If not collated, classes just map to itself.
        positive_class (str): Name of "good" class.
        extract_info_from_file_name: If set to True, a regex expression will extract category,
            sample, revolution and trigger number from file name
        regex_expression_folder_path: regex expression to information from filename
        regex_expression_file_name: regex expr to extact category information from\
            image name
        benchmark (list): Benchmark [false reject rate, detection rate] of classic vision 
            detection for this camera station. Will be plotted in ROC Curve.
    
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
    extract_info_from_file_name: bool
    regex_expression_file_name: str
    benchmark: list

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.import_file_path = Path(self.import_file_path)
        if self.model_classes == None:
            class_json_path = self.import_file_path.parent / 'final' / 'classes.json'
            with open(class_json_path) as f:
                self.model_classes =  json.load(f)
            self.model_classes_to_index = {j:i for i,j in enumerate(self.model_classes)}
        self.image_name_column ='img'
        self.score_column = 'scores'
        self.truth_column = 'image_truth'

class ExportParams:
    export_path: Path

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.export_path = Path(self.export_path)

def load_params(path: str, params_type: str):
    params_path = Path(path)
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    if params_type == 'import_params':
        return ImportParams(**params[params_type])
    elif params_type == 'export_params':
        return ExportParams(**params[params_type])

def extract_info(
    df: pd.DataFrame,
    import_params: ImportParams,
    extract_info_from_file_name: bool = False
)-> pd.DataFrame:
    """Extracts info from folder names, file names and scores of data frame.

    Names of folder and images contain information of the category, sample, 
    revolution, and sequence of each image. These information are extractec 
    via regular expressions.

    Args:
        df (pd.DataFrame): _description_
        image_folder_column (str): _description_
        image_name_column (str): _description_
        score_column (str): _description_
        category_map (dict): _description_

    Returns:
        pd.DataFrame: _description_
    """    
    def get_file_infos(file_path):
        p = Path(file_path)
        file_name = p.name
        return list(re.search(import_params.regex_expression_file_name, file_name).groups())

    def get_score_list(scores: str):
        return list(map(float, scores[1:-1].split()))

    def get_ground_truth(file_name):
        return Path(file_name).parent.name

    df_columns = [
        'image_name',
        'category',
        'sample', 
        'revolution',
        'trigger',
        'image_truth'
        ]

    df_out = pd.DataFrame(columns=df_columns)
    df_out['image_name'] = df[import_params.image_name_column]
    if extract_info_from_file_name:
        img_infos = df_out['image_name'].apply(get_file_infos)
        df_out['category'] = img_infos.apply(lambda x: x[0])
        df_out['sample'] = img_infos.apply(lambda x: int(x[1]))
        df_out['revolution'] = img_infos.apply(lambda x: int(x[2]))
        df_out['trigger'] = img_infos.apply(lambda x: int(x[3]))
    
    scores = df[import_params.score_column].apply(get_score_list)
    for n, category in enumerate(import_params.model_classes):
        df_out[category] = scores.apply(lambda x: x[n])
    df_out['image_truth'] = df_out['image_name'].apply(get_ground_truth)

    return df_out

def load_results(
    path: Path, 
    import_params: ImportParams
)-> pd.DataFrame:
    df_results = pd.read_csv(
        path, 
        names = [
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
