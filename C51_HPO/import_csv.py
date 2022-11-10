import pandas as pd
import re
from typing import List


def extract_info(file_name):
    info = re.search(r"___(.+)_seq_(\d+)_[a-zA-Z]+_(\d+)(\.[a-zA-Z]+)$", file_name)
    # some files are wrongly named. Another regex is needed for these files
    if not info:
        info = re.search(r"___(.+)_seq_(\d+)_[a-zA-Z]+_(\d+)[\w\s\(\)]+(\.[a-zA-Z]+)$", file_name)
    return list(info.groups())

def get_prediction(scores_str: str):
    score_list = list(map(float, re.findall(r"([\w\.\-\+]+)", scores_str)))
    return score_list.index(max(score_list))

def load_test_scores(path):
    results = pd.read_csv(path, names = ['img','scores','truth'])

    img_infos = results.img.apply(extract_info)
    results['sample'] = img_infos.apply(lambda x: x[0])
    results['revolution'] = img_infos.apply(lambda x: int(x[1]))
    results['sequence'] = img_infos.apply(lambda x: int(x[2]))
    results['prediction'] = results.scores.apply(get_prediction)

    return results

def main():
    path = './C51_HPO/test_scores.csv'
    test_scores_df = load_test_scores(path)
    test_scores_df.head()

if __name__ == '__main__':
    main()