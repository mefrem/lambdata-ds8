'''
lambdata - a collection of data science helper functions
'''

import pandas as pd
import numpy as np

# sample code
ones = pd.DataFrame(np.ones(10))
zeros = pd.DataFrame(np.zeros(50))

# sample functions
def increment(x):
    return x + 1