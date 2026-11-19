import hashlib

import numpy as np

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    df["bucket"] = df[id_column] % 100
    df["sample"] = np.where(
        df["bucket"].between(1, 100*training_frac),
        "train",
        "test",
    )
    return df