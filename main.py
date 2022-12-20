import pandas as pd
import numpy as np
from linear_regression import LinearRegression

print('linear_regression')

check = LinearRegression(pd.DataFrame({'f1': [1, 2, 3, 4, 5], 'f2': [2, 3, 4, 5, 6]}),
                         pd.DataFrame({'t':[3, 5, 7, 9, 11]}))
check.x = check.add_bias()