import pandas as pd
import re
from typing import List
from pathlib import Path
import yaml


class ImportParams:
    import_file_path: Path
    image_folder_column: str
    image_name_column: str
    score_column: str
    category_map: str

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def extract_info(df, image_folder_column, image_name_column, score_column, category_map):
    def get_category_from_folder_path(folder_path):
        folder_name = re.search(r"([^\\]+)$", folder_path).group()
        # second re.search not needed if numbers are removed from folder name
        return re.search(r"^([^\d]+)\s", folder_name).group(1)
    
    def get_file_infos(file_name):
        return list(re.search(r"(\d+)_(\d+)_(\d+).\w+$", file_name).groups())

    def get_prediction(prediction):
        if prediction.lower().lstrip() == "true":
            return 1
        elif prediction.lower().lstrip() == "false":
            return 0
        else:
            raise ValueError('Please check if prediction values are correctly formated!'\
                'Only "TRUE" and "FALSE" are allowed')

    def get_ground_truth(category, category_map):
        return category_map[category]

    df_columns = [
        'image_folder', 
        'image_name',
        'category',
        'sample', 
        'revolution',
        'sequence',
        'prediction',
        'truth'
        ]

    df_out = pd.DataFrame(columns=df_columns)
    df_out['image_folder'] = df[image_folder_column]
    df_out['image_name'] = df[image_name_column]
    df_out['category'] = df[image_folder_column].apply(get_category_from_folder_path)

    img_infos = df_out['image_name'].apply(get_file_infos)
    df_out['sample'] = img_infos.apply(lambda x: int(x[0]))
    df_out['revolution'] = img_infos.apply(lambda x: int(x[1]))
    df_out['sequence'] = img_infos.apply(lambda x: int(x[2]))
    df_out['prediction'] = df[score_column].apply(get_prediction)
    df_out['truth'] = df_out.category.apply(get_ground_truth, args=(category_map,))

    return df_out

def load_results(path, image_folder_column, image_name_column, score_column, category_map):
    df_results = pd.read_csv(path, header=1)
    df_results = extract_info(df_results, image_folder_column, image_name_column, score_column, category_map)
    return df_results

def convert_test_scores_to_sample_scores(scores_df):
    df = scores_df.copy()
    grouped_df = df.groupby(['category','sample','revolution'])
    predictions = grouped_df['prediction'].max()
    sample_truths = grouped_df['truth'].first()
    max_sample_scores = pd.concat([predictions, sample_truths], axis=1)
    return max_sample_scores

def main(path, image_folder_column, image_name_column, score_column, category_map):
    test_scores_df = load_results(path, image_folder_column, image_name_column, score_column, category_map)
    return test_scores_df

if __name__ == '__main__':
    params_path = Path('./SVIM_export/params.yml')
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    import_params = ImportParams(**params['import_params'])
    
    main(
        path=import_params.import_file_path,
        image_folder_column=import_params.image_folder_column,
        image_name_column=import_params.image_name_column,
        score_column=import_params.score_column, 
        category_map=import_params.category_map)