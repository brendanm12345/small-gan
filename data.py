import pandas as pd
import numpy as np
import json
import os
import shutil

# Load the original dataset
df = pd.read_csv('./ocular-disease-recognition-odir5k/full_df.csv')

raw_data = df.drop(columns=['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O', 'labels'])

raw_data.head()

targets = np.array(raw_data["target"].apply(lambda x: json.loads(x)).tolist())

print("Shape of the targets:", targets.shape)

classes = {0: "Normal",
           1: "Diabetes",
           2: "Glaucoma",
           3: "Cataract",
           4: "Age related Macular Degeneration",
           5: "Hypertension",
           6: "Pathological Myopia",
           7: "Other diseases/abnormalities"
           }

data = np.sum(targets, axis=0)

classes_names = list(classes.values())
values = list(data)

raw_data["class_name"] = np.argmax(targets, axis=1).tolist()
raw_data["class_name"] = raw_data["class_name"] .replace(classes)

raw_data.head()