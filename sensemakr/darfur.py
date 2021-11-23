import pandas as pd
import os

path=os.path.join(os.path.dirname(__file__), '../data/darfur.csv')

def load_darfur():
    return pd.read_csv(path)
