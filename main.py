import pandas as pd
import numpy as np
from numpy.linalg import inv, matrix_power
from linear_regression import LinearRegression

x = pd.DataFrame({'f1': [1, 2, 3, 4, 5], 'f2': [2, 3, 4, 5, 6]})
y = pd.DataFrame({'t':[3, 5, 7, 9, 11]})
check = LinearRegression(x, y)

x = check.add_bias(x)
y = check.y
y_ols = check.predict(check.ols())
y_gd = check.predict(check.gradient_descent())
y_sgd = check.predict(check.stochastic_gradient_descent())
y_ridge = check.predict(check.stochastic_gradient_descent())
print(check.mse(y_ols), check.mse(y_gd),
      check.mse(y_sgd), check.mse(y_ridge))
#print(check.x[5 - 3: 5])

