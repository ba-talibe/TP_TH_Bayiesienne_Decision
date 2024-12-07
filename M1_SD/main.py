from algorithms import PerceptronNaif
import numpy as np  



perceptron = PerceptronNaif()

x = np.array([-2 , -1, 0, 1, 2])
y = np.array([1, 0, 0, 0, 1])

x = x.reshape(-1,1)


perceptron.fit(x, y, 5000)

print(perceptron.predictor(x))