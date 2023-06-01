import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import streamlit as st
from scipy import stats 
from sklearn.linear_model import LinearRegression

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed
X, y = sklearn.datasets.make_regression(n_samples = 10000, 
                       n_features=1, 
                       n_informative=1, 
                       noise=20,
                       random_state=2019)
st.map(X, y)