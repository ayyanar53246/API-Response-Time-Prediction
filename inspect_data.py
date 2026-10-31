import pandas as pd

df = pd.read_csv("access_logs.csv")

print("DATASET LOADED SUCCESSFULLY!")
print("Dataset shape:", df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nMISSING VALUES:")
print(df.isnull().sum())