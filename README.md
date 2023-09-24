# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import the required library and read the dataframe.

2. Write a function computeCost to generate the cost function.

3. Perform iterations og gradient steps with learning rate.

4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Shalini.K
RegisterNumber: 212222240095
# Import required package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
data
data.shape
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Vs Prediction")
def computeCost(X,y,theta):
    m=len(y)
    h=X.dot(theta)
    square_err=(h-y)**2

    return 1/(2*m) * np.sum(square_err)
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)
theta.shape
y.shape
X.shape
def gradientDescent(X,y,theta,alpha,num_iters):
  
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions - y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta, J_history
  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
plt.plot(J_history)
plt.xlabel("Iternations")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Polpulation of City (10,000s)")
plt.ylabel("Profit (10,000s)")
plt.title("Profit Prediction")
def predict(x,theta):
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
### Read CSV FILE :
![1](https://github.com/shalinikannan23/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118656529/c3890f4a-241e-42ea-a7c8-9b7980d000ff)
### Dataset shape :
![2](https://github.com/shalinikannan23/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118656529/c6e1aec4-cfc2-4f19-a113-96a9a15a00f1)
### Profit Vs Prediction graph:
![3](https://github.com/shalinikannan23/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118656529/5033cb3f-53d3-42a5-a78a-2a0323e1dbba)
### x,y,theta value:
![4](https://github.com/shalinikannan23/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118656529/c57997fb-fd64-4d3e-a341-bd266b6dc9a9)
### Gradient descent:
![6](https://github.com/shalinikannan23/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118656529/4d3b689e-7796-4682-9fb5-3509db4c9c98)
### Cost function using Gradient Descent Graph:
![7](https://github.com/shalinikannan23/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118656529/d864a41e-71b9-4d05-95a3-733fe9f09f72)
### Profit Prediction Graph:
![8](https://github.com/shalinikannan23/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118656529/caa5a204-03ad-45d2-9a69-1d69d665673b)
### Profit Prediction:
![9](https://github.com/shalinikannan23/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118656529/f6b88cdc-5266-4940-bc59-10c8ce0b7f50)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
