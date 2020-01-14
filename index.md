## How to save the gradient and optimize learning with Replicas

One of the main difficulties encountered in the learning process of deep neural networks is the so called "Vanishing Gradient" problem.
We will show that, having some knowledge of how the learning works, one can easily improve the learning rate of the first layers.

### The Space of Solutions

The standard procedure for supervised learning has at its core the objective of minimizing the error, or cost function:

![](https://latex.codecogs.com/gif.latex?J%28%5Csigma%29%3D%5Cmathbb%7BE%7D_%7B%28x%2Cy%29%5Csim%20p_%7Bdata%7D%7D%5BL%28f%28x%3B%5Csigma%29%2Cy%29%5D)

Where ![](https://latex.codecogs.com/gif.latex?x) and ![](https://latex.codecogs.com/gif.latex?y) are our imputs and expected outputs, and ![](https://latex.codecogs.com/gif.latex?%5Csigma) is our configuraton of weights ![](https://latex.codecogs.com/gif.latex?w%5Cin%5Csigma) (see[1] for a detailed review).
The N weights span a N dimensional space, similar to ![](https://latex.codecogs.com/gif.latex?%5Cmathbb%7BR%7D%5EN), and ![](https://latex.codecogs.com/gif.latex?%5Csigma) is a point in it.
What's going on when a NN adjusts its weights according to a Stochastic Gradient Descent algorithm?
What's really happening is that the the N dimensional space of all the weights is being searched for a minima of the cost function.
All the optimization algorithms aim to search through this space, in order to find the ![](https://latex.codecogs.com/gif.latex?%5Csigma) yielding the minimum cost ![](https://latex.codecogs.com/gif.latex?J%28%5Csigma%29).

The space of solutions is a highly non trivial object, and this is why we need powerfull instruments coming from Statistical Physics (Thermodynamics). In Statistical Physics, the canonical ensemble describes the equilibrium (i.e., long-time limit) properties of a stochastic process in terms of a probability distribution over the configurations ![](https://latex.codecogs.com/gif.latex?%5Csigma) of the system:

![](https://latex.codecogs.com/gif.latex?P%28%5Csigma%3B%5Cbeta%29%3DZ%28%5Cbeta%29%5E%7B-1%7D%5Cexp%7B%28-%5Cbeta%20E%28%5Csigma%29%29%7D)

Where E (σ) is the energy of the configuration, β is the inverse of the temperature, and the normalization factor Z (β) is called the partitionfunction and can be used to derive all equilibrium properties.
 This distribution is thus defined whenever a function E (σ) is provided, and indeed it can be studied and can provide insight even when the system under consideration is not a physical system. In particular, it can be used to describe interesting properties of optimization problems, in which E (σ) has the role of a cost function that one wishes to minimize; in these cases, one is interested in the limit
β → ∞, which corresponds to assigning a uniform weight over the global minima of the energy function. This kind of description is at the core of the well-known Simulated Annealing algorithm [2].

It has been shown in a seminal paper [3] that the space of solutions contains global minima, which are rare thus hard to find, and a multitude of local minima, which are found in very dense clusters.

This motivated us to introduce a different measure, which ignores isolated solutions and enhances the statistical weight of large, accessible regions of solutions:

![](https://latex.codecogs.com/gif.latex?P%28%5Csigma%3B%5Cbeta%2Cy%2C%5Cgamma%29%3DZ%28%5Cbeta%2Cy%2C%5Cgamma%29%5E%7B-1%7De%5E%7By%5CPhi%28%5Csigma%2C%5Cbeta%2C%5Cgamma%29%7D)

Here, y is a parameter that has the formal role of the inverse temperature and Φ(σ,γ,β) is a “local entropy”:

![](https://latex.codecogs.com/gif.latex?%5CPhi%28%5Csigma%2C%5Cbeta%2C%5Cgamma%29%3D%5Clog%7B%5Csum_%7B%5C%7B%5Csigma%27%5C%7D%7D%5E%7B%20%7De%5E%7B-%5Cbeta%20E%28%5Csigma%27%29-%5Cgamma%20d%28%5Csigma%27%2C%5Csigma%29%7D%7D)

 d(·,·) being some monotonically increasing function of the distance between configurations.
In the limit β →∞, this expression reduces (up to an additive constant) to a “local entropy”: It counts the number of minima of the energy, weighting them (via the parameter γ) by the distance from a reference configuration σ. Therefore, if y is large, only the configurations σ that are surrounded by an exponential number of local min- ima will have a nonnegligible weight. By increasing the value of γ, it is possible to focus on narrower neighborhoods around σ, and at large values of γ the reference σ will also with high probability share the properties of the surrounding minima.

From standart Statistical Mechanics, we can retrieve our cost using 

![](https://latex.codecogs.com/gif.latex?%5Cleft%5Clangle%20E%20%5Cright%5Crangle%3D-%5Cfrac%7B%5Cpartial%20%7D%7B%5Cpartial%20%5Cbeta%7DlnZ)

To evaluate Z, we can rewrite it as

![](https://latex.codecogs.com/gif.latex?Z%28%5Cbeta%2Cy%2C%5Cgamma%29%3D%5Csum_%7B%5C%7B%5Csigma%5E*%5C%7D%7D%5E%7B%20%7De%5E%7By%5CPhi%28%5Csigma%5E*%2C%5Cbeta%2C%5Cgamma%29%7D%3D%5Csum_%7B%5C%7B%5Csigma%5E*%5C%7D%7D%5Csum_%7B%5C%7B%5Csigma%5Ea%5C%7D%7D%5E%7B%20%7D%20e%5E%7B-%5Cbeta%5Csum_%7Ba%3D1%7D%5E%7By%7DE%28%5Csigma%5Ea%29-%5Cgamma%5Csum_%7Ba%3D1%7D%5E%7By%7Dd%28%5Csigma%5E*%2C%5Csigma%5Ea%29%7D)

This partition function describes a system of y + 1 interacting replicas of the system, one of which acts as reference while the remaining y are identical, subject to the energy E (σ^a) and the interaction with the reference σ* .Studying the equilibrium statistics of this system and tracing out the replicas σ a is equivalent to studying the original model. This provides us with a very simple and general scheme to direct algorithms to explore robust, accessible regions of the energy landscape: replicating the model, adding an interaction term with a referencecon figuration, and running the algorithm over the resulting extended system.
In fact, in mostcases, we can further improve on this scheme by tracing out the reference instead, which leaves us with a system
of y identical interacting replicas

![](https://latex.codecogs.com/gif.latex?Z%28%5Cbeta%2Cy%2C%5Cgamma%29%3D%5Csum_%7B%5C%7B%5Csigma%5Ea%5C%7D%7De%5E%7B-%5Cbeta%5Csum_%7Ba%3D1%7D%5Ey%20E%28%5Csigma%5Ea%29&plus;A%28%5C%7B%5Csigma%5Ea%5C%7D%2C%5Cbeta%2C%5Cgamma%29%7D)

![](https://latex.codecogs.com/gif.latex?A%28%5C%7B%5Csigma%5Ea%5C%7D%2C%5Cbeta%2C%5Cgamma%29%3D-%5Cfrac%7B1%7D%7B%5Cbeta%7D%5Clog%7B%5Csum_%7B%5Csigma%5E*%7D%20e%5E%7B-%5Cgamma%5Csum_%7Ba%3D1%7D%5Ey%20d%28%5Csigma%5E*%2C%5Csigma%5Ea%29%7D%7D)


### A new learning rule

All of this can be used to evaluate a new cost function J := \<E\> and thus we can compute our gradient, updating the weights w <- w-c(dE/dw). Our new cost function will be

![](https://latex.codecogs.com/gif.latex?H%28%7BW%5Ea%7D%29%3D%5Csum_%7Ba%3D1%7D%5Ey%20E%28W%5Ea%29-%5Cfrac%7B1%7D%7B%5Cbeta%7D%20%5Csum_%7Bj%3D1%7D%5EN%20%5Clog%7B%28e%5E%7B-%5Cfrac%7B%5Cgamma%7D%7B2%7D%5Csum_%7Ba%3D1%7D%5Ey%28W%5Ea_j-1%29%5E2%7D&plus;e%5E%7B-%5Cfrac%7B%5Cgamma%7D%7B2%7D%5Csum_%7Ba%3D1%7D%5Ey%28W%5Ea_j&plus;1%29%5E2%7D%29%7D)

and therefore the gradient just has an additional term:

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20H%7D%7B%5Cpartial%20W%5Ea_i%7D%28%5C%7BW%5Eb%5C%7D%29%3D%5Cfrac%7B%5Cpartial%20E%7D%7BW_i%7D%28W%29%5Cbigg%5Crvert_%7BW%3DW%5Ea%7D%20-%5Cfrac%7B%5Cgamma%7D%7B%5Cbeta%7D%28%5Ctanh%7B%28%5Cgamma%5Csum_%7Bb%3D1%7D%5Ey%20W%5Eb_i%29%20-%20W%5Ea_i%7D%29)

Our new learning rule thus will be:

![image](rep.jpg)

Let's test it!\\
Here are some graphs of the cost and the learning rate of the layers for a NN with two layers
![image](learning1.png)

Here we can compare the results with the old learning rule (on the left) and with the new one (on the right)\\
The Cost is plotted in orange, the learning rate of the second layer is blue, and the first is red.\\
With the old rule, the first layer (red) learns a couple of orders less than the first (blue), whereas with the new rule,
one can clearly see how the first layer (red) improves its learning by at least an order

### Bibliography and further reading

[1] I.Goodfellow, Y.Bengio and A.Courville *Dep learning*, MIT press, 2017

[2] J.S. Yedidia, W.T. Freeman, and Y. Weiss. Constructing free-energy approximations and generalized belief
propagation algorithms. Information Theory, IEEE Transactions on, 51(7):2282–2312, 2005

[3] C.Baldassi, A.Ingrosso, C.Lucibello, L.Saglietti, and R.Zecchina
Phys. Rev. Lett. 115, 128101 – 2015

[4] C.Baldassi, C.Borgs, J.T. Chayes, A.Ingrosso, C.Lucibello, L.Saglietti, R.Zecchina
Proceedings of the National Academy of Sciences Nov 2016, 113 (48) E7655-E7662;

### Code

Here's the code I used for the proof of concept of the new algorithm.
It uses **only 2 replicas**, training them simultaniously, giving them different stochastic batch imputs.\\
Once the Forward is done, the weights are updatet according to the new rule.\\
Despite the small numbers of replicas, the improvement is already huge.
Adding more replicas will result in an even bigger increase.

```ruby

"""
Created on Wed Jun 19 11:19:10 2019

@author: Fabbricatore
"""

# No fancy libraries

import numpy as np
import matplotlib.pyplot as plt

zz=np.load("x_train.npy")
tt=np.load("t_train.npy")

# N     = size of Training Set
# B     = batch size
# D_in  = dimension of input vectors
# H     = neurons in Hidden Layer
# D_out = dimension of output vectors
N, B, D_0, D_in, H, D_out = 60000, 50, 784, 100, 30, 10

# Replica number
R=7

# List for the Network Replica
nets = list()

# Learning rates, the second one in for the replica coupling
learning_rate = 1e-7
learning_rate2 = 1e-3

# Number of learnng epochs
times=100

# Initializing plots
a=np.zeros(times)
b=np.zeros(times)
c=np.zeros(times)
ar=np.zeros(times)
br=np.zeros(times)
d=np.zeros(times)
dr=np.zeros(times)

# Initializing random training set
xx = np.load("x_train.npy")/255
yy = np.random.randn(N, D_out)

for i in range(N):
    yy[i]=(np.arange(10)==tt[i]).astype(np.int)*1000


# NETWORK class
class net :
    
    def __init__(self, D_0, D_in, H, D_out):
        self.D_0    =   D_0
        self.D_in   =   D_in
        self.D_out  =   D_out
        self.H      =   H
        self.w0     =   np.random.randn(self.D_0,self.D_in)
        self.w1     =   np.random.randn(self.D_in,self.H)
        self.w2     =   np.random.randn(self.H,self.D_out)
        
    def work (self,x,y):
        
        # Forward
        h0 = x.dot(self.w0)
        h0_relu = np.maximum(h0, 0)
        h = h0_relu.dot(self.w1)
        h_relu = np.maximum(h, 0)
        self.y_pred = h_relu.dot(self.w2)

        # Bacward
        grad_y_pred = 2.0 * (self.y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(self.w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = h0_relu.T.dot(grad_h)
        grad_h0_relu = grad_h.dot(self.w1.T)
        grad_h0 = grad_h0_relu.copy()
        grad_h0[h0 < 0] = 0
        grad_w0 = x.T.dot(grad_h0)
    
        # Update weights
        self.w0 -= learning_rate * grad_w0
        self.w1 -= learning_rate * grad_w1
        self.w2 -= learning_rate * grad_w2

        
        self.gr0  =  abs( np.mean( learning_rate * grad_w0) )
        self.gr1  =  abs( np.mean( learning_rate * grad_w1) )
        self.gr2  =  abs( np.mean( learning_rate * grad_w2) )
        
        return self.y_pred
        
    def grR(self):
        
        # Replicas update
        self.w0 -= learning_rate2 * (gradR()[0] - self.w0)
        self.w1 -= learning_rate2 * (gradR()[1] - self.w1)
        self.w2 -= learning_rate2 * (gradR()[2] - self.w2)

        
        self.gr0R  =  abs( np.mean( learning_rate2 * gradR()[0]) )
        self.gr1R  =  abs( np.mean( learning_rate2 * gradR()[1]) )
        self.gr2R  =  abs( np.mean( learning_rate2 * gradR()[2]) )
        #print(self.gr1, self.gr1R)
        
 # Create R Replicas
for i in range(R):
    nets.append(net(D_0, D_in, H, D_out))        
   
# Gradient for the Replica coupling
def gradR():
    gr0, gr1, gr2 =  np.zeros([D_0, D_in]), np.zeros([D_in, H]), np.zeros([H, D_out])
    for i in range(R):
        gr0 += nets[i].w0/R
        gr1 += nets[i].w1/R
        gr2 += nets[i].w2/R
    return gr0, gr1, gr2
    
    
def loss(y_pred, y): 
    return np.square(y_pred - y).sum()
    
############################   Let's test it   ######################################## 


for k in range(1): # Repeat the test 100 times 
    
    for i in range(R):
        nets[i].w1 = np.random.randn(D_in,H) # Randomly initialize weights every time
        nets[i].w2 = np.random.randn(H,D_out)
        nets[i].w0 = np.random.randn(D_0,D_in)
        
    for j in range(times): 
        
        r = np.random.randint(N, size = B) #Randomly pick from Data set
        x = xx[r]
        y = yy[r]
        
        for i in range(R):  # let all replicas work
            nets[i].work(x,y)
            
        for i in range(R):
            nets[i].grR()   # Update weights with new rule
            
        a[j]    +=  nets[0].gr1  # Save averaged results
        b[j]    +=  nets[0].gr2
        d[j]    +=  nets[0].gr0
        ar[j]   +=  nets[0].gr1R
        br[j]   +=  nets[0].gr2R
        dr[j]   +=  nets[0].gr0R
        c[j]    +=  loss(y, nets[0].y_pred)
    
    
plt.plot(a*50,'r')
plt.plot(b)
plt.plot(c/100000000)
plt.ylabel('rearning rate')
plt.xlabel(' iterations ')
plt.show()


fig = plt.figure(figsize = (12,5))

ax=fig.add_subplot(3,2,1).plot(b)
plt.title('w3')
fig.add_subplot(3,2,3).plot(a,'r')
plt.title('w2')
fig.add_subplot(3,2,5).plot(d,'g')
plt.title('w3')
fig.add_subplot(3,2,2).plot(br)
plt.title('w1 R')
fig.add_subplot(3,2,4).plot(ar,'r')
plt.title('w2 R')
fig.add_subplot(3,2,6).plot(dr,'g')
plt.title('w3 R')


# Printing numbers
fag = plt.figure(figsize=(15,3))

for i in range(10):
    axs = fag.add_subplot(1,10,i+1) # (row, col, number inside (1<x<row*col))
    axs.imshow(np.reshape(x[i,:],(28,28)), cmap="gray")
    axs.set_xlabel(np.argmax(nets[0].y_pred[i]))
    axs.set_title(np.argmax(y[i]))
    axs.set_xticks([]); axs.set_yticks([])

```

