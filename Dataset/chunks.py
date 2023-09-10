import pandas as pd
import numpy as np



df=pd.read_csv("updated_dmoz_full_toplevel_imbalanced.csv")
chunk_size = 100000
df_chunks = np.array_split(df, len(df) // chunk_size + 1)

# Saving the smaller DataFrames in order
for i, chunk in enumerate(df_chunks):
    file_path = f"chunk_{i+1}.csv"
    chunk.to_csv(file_path, index=False)
    print(f"Chunk {i+1} saved as chunk_{i+1}.csv")