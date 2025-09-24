import numpy as np
import matplotlib.pyplot as plt

# Data
age = [1, 2, 3, 4, 5, 6]
cost = [7.55, 9.24, 10.74, 12.84, 15.66, 18.45]

variance_age = np.var(age)
variance_cost = np.var(cost)
covariance = np.cov(age, cost)[0][1]
linear_correlation = np.corrcoef(age, cost)[0][1]
# Linear regression
slope = covariance / variance_age
intercept = np.mean(cost) - slope * np.mean(age)
x_line = np.array([min(age), max(age)])
y_line = slope * x_line + intercept

# Print results
print("Variance of Age:", variance_age)
print("Variance of Cost:", variance_cost)
print("Covariance between Age and Cost:", covariance)
print("Linear Correlation between Age and Cost:", linear_correlation)

#print line equation
print("test", slope, intercept)
print(f"Linear Regression Line: Cost = {slope:.2f} * Age + {intercept:.2f}")

# plotting ( graphical representation )
plt.scatter(age, cost)
plt.title("Age vs Cost")
plt.xlabel("Age")
plt.ylabel("Cost")
plt.plot(x_line, y_line, color='red', label='Linear Regression Line')
plt.legend()
plt.grid()
plt.show()