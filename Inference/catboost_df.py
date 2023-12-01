import pandas as pd
import numpy as np
import os
import torch
from catboost import CatBoostClassifier

INPUTFILE=""
OUTPUTFILE=""

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

script_dir = os.path.dirname(os.path.realpath(__file__))
relative_path = '../Models/catboost_model.bin'
model_path = os.path.normpath(os.path.join(script_dir, relative_path))


clf = CatBoostClassifier()
clf.load_model(model_path)


def make_predictions_catboost_csv(INPUTFILE):

    df=pd.read_csv(INPUTFILE, encoding="cp1252")
    drop_column = ['image_unreachable','response_status_code','Title','Article','Text','Image_Name']
    df.drop(drop_column, axis=1, inplace = True)

    features = df.columns.tolist()
    features.remove('Url')

    X=df[features]
    predictions=clf.predict(X)
    print(predictions)

    df['Catboost_prediction'] = predictions


    OUTPUTFILE=INPUTFILE+"_processed_catboost.csv"
    df1=df[['Url','Catboost_prediction']]
    df1.to_csv(OUTPUTFILE, index=False, encoding='cp1252')

    print("Output saved in a csv named" ,OUTPUTFILE)


def main():

    INPUTFILE=sys.argv[1]
    make_predictions_catboost_csv(INPUTFILE)

   

if __name__ == "__main__":
    main()


