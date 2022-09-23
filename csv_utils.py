import pandas as pd

def convert_test_scores_csv_to_df(score_path):
    results = pd.read_csv(score_path, names = ['img','scores','truth'])
    results['good'] = results['scores'].apply(lambda x: float(x.split()[0][1:]))
    results['bad'] = results['scores'].apply(lambda x: float(x.split()[1][:-1]))
    results.drop(columns='scores', inplace=True)
    return results