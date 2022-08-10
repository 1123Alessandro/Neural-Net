import pandas as pd
import numpy as np

def readFile(file):
    return pd.read_csv(file)

# sample data generator
def div_data():
    rng = np.random.default_rng()
    l = list()
    for i in range(1000):
        a = rng.integers(1, 100)
        b = rng.integers(1, 100)
        l += [[a, b, a/b]]
    pd.DataFrame(l).to_csv('Div Data.csv', index=False)