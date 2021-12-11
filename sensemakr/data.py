"""
Description
------------
The module provides the example data of darfur based on a survey among Darfurian refugees in eastern Chad.

Reference
------------
Hazlett C (2019). “Angry or Weary? How Violence Impacts Attitudes toward Peace among
Darfurian Refugees.” Journal of Conflict Resolution.

Example
------------
>>> from sensemakr import data
>>> darfur = data.load_darfur()

"""
import pandas as pd
import os

path=os.path.join(os.path.dirname(__file__), 'data/darfur.csv')

def load_darfur():
    """ Load the darfur example data of the packages. """
    return pd.read_csv(path)
