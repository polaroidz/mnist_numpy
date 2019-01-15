# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

dataset = pd.read_csv('./train.csv')

X, y = dataset.iloc[:,1:], dataset.iloc[:,:1]

del dataset

