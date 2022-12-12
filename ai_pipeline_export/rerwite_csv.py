from pathlib import Path

import_path = r'C:\Users\1699\tmp\Cilag_experiments\c61_collated\model_20221208_mnp50_62_edge_length\val_scores.csv'
export_path = r'C:\Users\1699\tmp\Cilag_experiments\c61_collated\model_20221208_mnp50_62_edge_length\val_scores_converted.csv'

file1 = open(import_path, 'r')

Lines = file1.readlines()

new_lines = [Lines[0],]

for line in Lines[1:]:
    if line[0:5] == '/data':
        new_lines.append(line)
    else:
        new_lines[-1] = new_lines[-1][:-1] + line

file2 = open(export_path, 'w')
file2.writelines(new_lines)
file2.close()