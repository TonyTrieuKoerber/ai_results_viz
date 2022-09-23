from pathlib import Path
import shutil
import pandas as pd
import csv_utils

def make_directories(path):
    destination_dir = path.parent.parent / "1_statistics"
    false_positives_dir = destination_dir / "false_positives"
    false_negative_dir = destination_dir / "false_negatives"
    false_positives_dir.mkdir()
    false_negative_dir.mkdir()
    return destination_dir, false_positives_dir, false_negative_dir

def save_false_positives_and_negatives(score_path, destination_dir, threshold):
    results = csv_utils.convert_test_scores_csv_to_df(score_path)
    
    false_positives = results[(results.truth == 1) & (results.bad < threshold)]
    false_negatives = results[(results.truth == 0) & (results.bad >= threshold)]
        
    false_positives.to_csv(destination_dir / "false_positives.csv", index=False)
    false_negatives.to_csv(destination_dir / "false_negatives.csv", index=False)
    return false_negatives, false_positives

def copy_false_images(data_path, img_path, destination_dir):
    file_path = Path(img_path)
    src = data_path.joinpath(img_path)
    dst = destination_dir / file_path.name
    shutil.copy(src, dst)

def main(score_path, data_path, threshold):
    destination_dir, false_positives_dir, false_negative_dir = make_directories(score_path)
    df_false_neg, df_false_pos = save_false_positives_and_negatives(score_path, destination_dir, threshold)
    for img_path in df_false_pos['img']:
        copy_false_images(data_path, img_path, false_positives_dir)
    for img_path in df_false_neg['img']:
        copy_false_images(data_path, img_path, false_negative_dir)



if __name__ == '__main__':
    score_file_path = Path(r"/home/tonytrieu/repositories/ai_results/Testing/model/test_scores.csv")   
    img_src_path = Path(r"/home/tonytrieu/datasets/AZ/1_manually_split_data_model/1_Manually_split_dataset_2_binary/AOI_5_test_complete_good_samples/test")
    threshold = 0.5

    main(score_file_path, img_src_path, threshold)

    
    
    
    
