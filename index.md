## How to save the gradient and optimize learning with Replicas

One of the main difficulties encountered in the learning process of deep neural networks is the so called "Vanishing Gradient" problem.
We will show that, having some knowledge of how the learning works, one can easily improve the learning rate of the first layers.

### The Space of Solutions

The standard procedure for supervised learning has at its core the objective of minimizing the error, or cost function:

![](https://latex.codecogs.com/gif.latex?J%28%5Csigma%29%3D%5Cmathbb%7BE%7D_%7B%28x%2Cy%29%5Csim%20p_%7Bdata%7D%7D%5BL%28f%28x%3B%5Csigma%29%2Cy%29%5D)

Where ![](https://latex.codecogs.com/gif.latex?x) and ![](https://latex.codecogs.com/gif.latex?y) are our imputs and expected outputs, and ![](https://latex.codecogs.com/gif.latex?%5Csigma) is our configuraton of weights ![](https://latex.codecogs.com/gif.latex?w%5Cin%5Csigma) (see[1] for a detailed review).\
The N weights span a N dimensional space, similar to ![](https://latex.codecogs.com/gif.latex?%5Cmathbb%7BR%7D%5EN), and ![](https://latex.codecogs.com/gif.latex?%5Csigma) is a point in it.
What's going on when a NN adjusts its weights according to a Stochastic Gradient Descent algorithm?\
What's really happening is that the the N dimensional space of all the weights is being searched for a minima of the cost function.
All the optimization algorithms aim to search through this space, in order to find the ![](https://latex.codecogs.com/gif.latex?%5Csigma) yielding the minimum cost ![](https://latex.codecogs.com/gif.latex?J%28%5Csigma%29).

The space of solutions is a highly non trivial object, and this is why we need powerfull instruments coming from Statistical Physics (Thermodynamics). In Statistical Physics, the canonical ensemble describes the equilibrium (i.e., long-time limit) properties of a stochastic process in terms of a probability distribution over the configurations ![](https://latex.codecogs.com/gif.latex?%5Csigma) of the system:

![](https://latex.codecogs.com/gif.latex?P%28%5Csigma%3B%5Cbeta%29%3DZ%28%5Cbeta%29%5E%7B-1%7D%5Cexp%7B%28-%5Cbeta%20E%28%5Csigma%29%29%7D)

Where E (σ) is the energy of the configuration, β an inverse temperature, and the normalization factor Z (β) iscalled the partitionfunction and can be used to derive all thermodynamic properties.
 This distribution is thus defined whenever a function E (σ) is provided, and indeed it can be studied and provide insight even when the system under consideration is not a physical system. In particular, it can be used to describe interesting properties of optimization problems, in which E (σ) has the role of a cost function that one wishes to minimize; in these cases, one is interested in the limit
β → ∞, which corresponds to assigning a uniform weight over the global minima of the energy function. This kind of description is at the core of the well-known Simulated Annealing algorithm (2).

It has been shown in a seminal paper (3) that the space of solutions contains global minima, which are rare thus hard to find, and a multitude of local minima. Despite our first desire, we should not aim for the global one, since they actually generalize worse.
Another property of the local minima is their appearance in clusters, rather than in a uniformly distributed configuration.

This motivated us to introduce a different measure, which ignores isolated solutions and enhances the statistical weight of large, accessible regions of solutions:

![](https://latex.codecogs.com/gif.latex?P%28%5Csigma%3B%5Cbeta%2Cy%2C%5Cgamma%29%3DZ%28%5Cbeta%2Cy%2C%5Cgamma%29%5E%7B-1%7De%5E%7By%5CPhi%28%5Csigma%2C%5Cbeta%2C%5Cgamma%29%7D)

Here, y is a parameter that has the formal role of an inverse temperature and Φ(σ,γ,β) is a “local free entropy”:

![](https://latex.codecogs.com/gif.latex?%5CPhi%28%5Csigma%2C%5Cbeta%2C%5Cgamma%29%3D%5Clog%7B%5Csum_%7B%5C%7B%5Csigma%27%5C%7D%7D%5E%7B%20%7De%5E%7B-%5Cbeta%20E%28%5Csigma%27%29-%5Cgamma%20d%28%5Csigma%27%2C%5Csigma%29%7D%7D)

 d(·,·) being some monotonically increasing function of the distance between configurations.
In the limit β →∞, this expression reduces (up to an additive constant) to a “local entropy”: It counts the number of minima of the energy, weighting them (via the parameter γ) by the distance from a reference configuration σ. Therefore, if y is large, only the configurations σ that are surrounded by an exponential number of local min- ima will have a nonnegligible weight. By increasing the value of γ, it is possible to focus on narrower neighborhoods around σ, and at large values of γ the reference σ will also with high probability share the properties of the surrounding minima.

From standart Statistical Physics, we can retrieve our cost using 

![](https://latex.codecogs.com/gif.latex?%5Cleft%5Clangle%20E%20%5Cright%5Crangle%3D-%5Cfrac%7B%5Cpartial%20%7D%7B%5Cpartial%20%5Cbeta%7DlnZ)

To evaluate Z, we can rewrite it as

![](https://latex.codecogs.com/gif.latex?Z%28%5Cbeta%2Cy%2C%5Cgamma%29%3D%5Csum_%7B%5C%7B%5Csigma%5E*%5C%7D%7D%5E%7B%20%7De%5E%7By%5CPhi%28%5Csigma%5E*%2C%5Cbeta%2C%5Cgamma%29%7D%3D%5Csum_%7B%5C%7B%5Csigma%5E*%5C%7D%7D%5Csum_%7B%5C%7B%5Csigma%5Ea%5C%7D%7D%5E%7B%20%7D%20e%5E%7B-%5Cbeta%5Csum_%7Ba%3D1%7D%5E%7By%7DE%28%5Csigma%5Ea%29-%5Cgamma%5Csum_%7Ba%3D1%7D%5E%7By%7Dd%28%5Csigma%5E*%2C%5Csigma%5Ea%29%7D)

This partition function describes a system of y + 1 interacting replicas of the system, one of which acts as reference while the remaining y are identical, subject to the energy E (σ^a) and the interaction with the reference σ* .Studying the equilibrium statistics of this system and tracing out the replicas σ a is equivalent to studying theo riginal model.This provides us with a very simple and general scheme to direct algorithms to explore robust, accessible regionsof the energy landscape: replicating the model,adding an interaction term with a referencecon figuration, and running the algorithm over the resulting extended system.
In fact,in mostcases,we can further improve on this scheme by tracing out the reference instead, which leaves us with a system
of y identical interacting replicas

![](https://latex.codecogs.com/gif.latex?Z%28%5Cbeta%2Cy%2C%5Cgamma%29%3D%5Csum_%7B%5C%7B%5Csigma%5Ea%5C%7D%7De%5E%7B-%5Cbeta%5Csum_%7Ba%3D1%7D%5Ey%20E%28%5Csigma%5Ea%29&plus;A%28%5C%7B%5Csigma%5Ea%5C%7D%2C%5Cbeta%2C%5Cgamma%29%7D)

![](https://latex.codecogs.com/gif.latex?A%28%5C%7B%5Csigma%5Ea%5C%7D%2C%5Cbeta%2C%5Cgamma%29%3D-%5Cfrac%7B1%7D%7B%5Cbeta%7D%5Clog%7B%5Csum_%7B%5Csigma%5E*%7D%20e%5E%7B-%5Cgamma%5Csum_%7Ba%3D1%7D%5Ey%20d%28%5Csigma%5E*%2C%5Csigma%5Ea%29%7D%7D)


### A new learning rule

All of this can be used to evaluate a new cost function J := \<E\> and thus we can compute our gradient, updating the weights w <- w-c(dE/dw)
Having in mind everything said above, we can develop a new rule to update the weights. After some tedious math, which I skip here, we come to a modified equation for the update rule.

![image](rep.jpg)

Let's test it!\\
Here are some graphs of the cost and the learning rate of the layers for a NN with two layers
![image](learning1.png)

Here we can compare the results with the old update rule (on the left) and with the new one (on the right)\\
The Cost is plotted in orange, the learning rate of the second layer is blue, and the first is red.\\
One can clearly see how the first layer improves its learning by at least an order

### Code

Here's the code I used to show the effect of the new algorithm.
It trains simultaniously 2 equal Networks, giving them different batch imputs.\\
Once the Forward is done, the weights are updatet according to the new rule.\\
Adding more replicas will result in an even bigger increase.

```ruby

import numpy as np
import matplotlib.pyplot as plt

#N is batch size; D_in is input dimension;
#H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

#Save values
a=np.zeros(50)
b=np.zeros(50)
c=np.zeros(50)

#Data library
xx = np.random.randn(10*N, D_in)
yy = np.random.randn(10*N, D_out)

#Create random input and output data
x1 = np.random.randn(N, D_in)
y1 = np.random.randn(N, D_out)

x2 = np.random.randn(N, D_in)
y2 = np.random.randn(N, D_out)

#Randomly pick from library

for i in range(N):
    rnd1 = np.random.randint(N)
    rnd2 = np.random.randint(N)
    x1[i] = xx[rnd1]
    y1[i] = yy[rnd1]
    x2[i] = xx[rnd2]
    y2[i] = yy[rnd2]

#Randomly initialize weights
w11 = np.random.randn(D_in, H)
w12 = np.random.randn(H, D_out)

w21 = np.random.randn(D_in, H)
w22 = np.random.randn(H, D_out)

learning_rate = 1e-6
eta2=1


for t in range(50):
    
    #Forward 1
    h = x1.dot(w11)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w12)

    #Loss 1
    loss = np.square(y_pred - y1).sum()
    
    #Backprop 1
    grad_y_pred = 2.0 * (y_pred - y1)
    grad_w12 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w12.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w11 = x1.T.dot(grad_h)
    
    ####################################
    
    #Forward 2
    h = x2.dot(w21)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w22)

    
    #Backprop 2
    grad_y_pred = 2.0 * (y_pred - y2)
    grad_w22 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w22.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w21 = x2.T.dot(grad_h)
    
    ###################################

    #Update weights
    w11 -= learning_rate * grad_w11-eta2*np.tanh((w11+w21)/2-w11)
    w12 -= learning_rate * grad_w12-eta2*np.tanh((w12+w22)/2-w12)
    
    w21 -= learning_rate * grad_w21-eta2*np.tanh((w11+w21)/2-w21)
    w22 -= learning_rate * grad_w22-eta2*np.tanh((w12+w22)/2-w22)
    
    if t<50 :
        a[t]=abs(np.mean(learning_rate * grad_w11-eta2*np.tanh((w11+w21)/2-w11)))
        b[t]=abs(np.mean(learning_rate * grad_w12-eta2*np.tanh((w12+w22)/2-w12)))
        c[t]=loss
    
    
    

plt.plot(a*50,'r')
plt.plot(b)
plt.plot(c/1000000000)
plt.ylabel('rearning rate')
plt.xlabel(' iterations ')
plt.show()

print(w11[1,1])

```

