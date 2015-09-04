# nn — a simple neural network

This is a result of my attempts to understand more about neural networks and how their underlying architecture
functions. It's a pure numpy implementation with almost no optimisation, it will run pretty slow unless you have a
decent CPU. 


## Benchmarks:
Accuracy of 96.8% on the MNIST Digits dataset using the follow architecture:

* `Hidden Layers = 400 x 300 x 200`
* `Activation = Sigmoid`
* `lmbda = 0.0`
* `mini_batch = 100 `
* `momentum = 'rmsprop' `
* `epochs = 300 `
* `eta = 1e-3 `
* `alpha = 0.25 `
* `p = 0.75`

**N.B. I make no assertions that this code will work outside of my testing environment. Feel free to fork, pull, 
whatever.**

Dependencies: Numpy, Python 3.4
