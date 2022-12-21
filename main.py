import pandas as pd
import numpy as np
from linear_regression import LinearRegression

x = pd.DataFrame({'f1': [1, 2, 3, 4, 5], 'f2': [2, 3, 4, 5, 6]})
y = pd.DataFrame({'t':[3, 5, 7, 9, 11]})
check = LinearRegression(x, y)
#check.x = check.add_bias()
x = check.add_bias(x)
print(check.predict())
