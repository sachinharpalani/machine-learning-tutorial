from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6],dtype=np.float64)
ys = np.array([5,4,6,5,6,7],dtype=np.float64)

#Formula for calculating slope and y-intercept
#slope = mean of (x) * mean of (y) - mean of (x*y)
#        -----------------------------------------
#        mean of (x)^2 - mean of (x^2)
#
#y-intercept = mean of (y) - slope * mean of (x)
#m and b are used to create model, use it on x data set and predict y

def best_fit_slope_and_intercept(xs,ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
          ((mean(xs)**2) - mean(xs**2)))

    b = mean(ys) - m*mean(xs)
    return m,b

m,b = best_fit_slope_and_intercept(xs,ys)

regression_line = [(m*x)+b for x in xs]

predict_x = 8
predict_y = (m*predict_x) + b

plt.scatter(xs,ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()
