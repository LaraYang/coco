# writes processed text responses from coco in a format expected by original BTM
import pandas as pd

coco_data = pd.read_csv("analyses_data/survey_hr_topic_modeling.csv")
with open('/ifs/projects/amirgo-identification/BTM/coco_data/pros.txt', 'w') as f:
    for item in coco_data['pros_cleaned'].tolist():
        f.write("%s\n" % item)
with open('/ifs/projects/amirgo-identification/BTM/coco_data/cons.txt', 'w') as f:
    for item in coco_data['cons_cleaned'].tolist():
        f.write("%s\n" % item)