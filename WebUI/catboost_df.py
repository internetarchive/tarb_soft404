import pandas as pd
import numpy as np
import os
import torch
from catboost import CatBoostClassifier


if torch.cuda.is_available():
    device_count=torch.cuda.device_count()
    print(f"Found {device_count} GPU")
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"Device {i}: {device_name}")
else:
    print("CUDA is not available on this system.")

print(torch.backends.cudnn.enabled)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

df=pd.read_csv("output1.csv", encoding="cp1252")
drop_column = ['image_unreachable','response_status_code','Title','Article','Text','Image_Name']
df.drop(drop_column, axis=1, inplace = True)

features = df.columns.tolist()
features.remove('Url')

X=df[features]

script_dir = os.path.dirname(os.path.realpath(__file__))
relative_path = '../Models/catboost_model.bin'
model_path = os.path.normpath(os.path.join(script_dir, relative_path))


clf = CatBoostClassifier()
clf.load_model(model_path)

predictions=clf.predict(X)
print(predictions)

df['Catboost_prediction'] = predictions



df1=df[['Url','Catboost_prediction']]
df1.to_csv('output1_cat.csv', index=False, encoding='cp1252')


