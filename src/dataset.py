import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path, test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
