import pandas as pd
import re
from typing import List
from pathlib import Path
import yaml


class ImportParams:
    """Class for importing import_params field of params.yml

    Attributes:
        import_file_path (Path): Path to results file.
        header: Row number to use as the column names, and the start of the data.
        image_folder_column (str): Column name of import file where to find the image folder.
        image_name_column (str): Column name of import file where to find the image folder.
        score_column (str): Column name of import file where to find the prediction score.
        regex_expression_folder_path: regex expression to extact category information\
            from folder path
        regex_expression_file_name: regex expr to extact category information from\
            image name
        category_map (dict): Dictionary to map categories to score.
    
    Example:
        >>> params_path = Path('./config/params.yml')
        >>> with open(params_path, 'r') as file:\
        >>>    params = yaml.safe_load(file)
        >>> import_params = ImportParams(**params['import_params'])

    """
    
    import_file_path: Path
    header: int
    image_folder_column: str
    image_name_column: str
    score_column: str
    regex_expression_folder_path: str
    regex_expression_file_name: str
    category_map: dict

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.import_file_path = Path(self.import_file_path)

def load_params(path: str):
    params_path = Path(path)
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return ImportParams(**params['import_params'])

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
    def get_category_from_image_name(folder_path):
        return re.search(r".+\/(.+)\/.+$", folder_path).group(1)
    
    def get_file_infos(file_name):
        return list(re.search(r"(\d+)_(\d+)_(\d+).\w+$", file_name).groups())

    def get_prediction(scores):
        scrores_list = list(map(float, scores[1:-1].split()))
        return scrores_list.index(max(scrores_list))

    df_columns = [
        'image_name',
        'category',
        'sample', 
        'revolution',
        'trigger',
        'prediction',
        'truth'
        ]

    df_out = pd.DataFrame(columns=df_columns)
    df_out['image_name'] = df[import_params.image_name_column]
    df_out['category'] = df_out['image_name'].apply(get_category_from_image_name)
    if extract_info_from_file_name:
        img_infos = df_out['image_name'].apply(get_file_infos)
        df_out['sample'] = img_infos.apply(lambda x: int(x[0]))
        df_out['revolution'] = img_infos.apply(lambda x: int(x[1]))
        df_out['trigger'] = img_infos.apply(lambda x: int(x[2]))
    df_out['prediction'] = df[import_params.score_column].apply(get_prediction)
    df_out['truth'] = df[import_params.truth_column]

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

def convert_test_scores_to_sample_scores(scores_df):
    df = scores_df.copy()
    grouped_df = df.groupby(['category','sample','revolution'])
    predictions = grouped_df['prediction'].max()
    sample_truths = grouped_df['truth'].first()
    max_sample_scores = pd.concat([predictions, sample_truths], axis=1)
    return max_sample_scores


if __name__ == '__main__':
    params_path = './ai_pipeline_export/params.yml'
    import_params = load_params(params_path)
    df_results = load_results(import_params.import_file_path, import_params)