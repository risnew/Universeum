import pandas as pd

from src import process_data

df = process_data(pd.read_csv("data/raw_data.csv"))
print(df.head())
