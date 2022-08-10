import numpy as np

# Activation functions
class ReLU():
    def forward(self, X):
        self.inputs = X
        self.outputs = np.maximum(0, X)
            
        return self.outputs

    def backward(self, dX):
        self.d_i = dX.copy()
        self.d_i[self.inputs < 0] = 0
        return self.d_i

class LReLU():
    def forward(self, X, a=0.01):
        self.inputs = X
        self.outputs = np.maximum(a*X, X)
        return self.outputs

    def backward(self, dX, a=0.01):
        self.d_i = np.ones_like(self.inputs, dtype=float)
        self.d_i[self.inputs < 0] = a
        # print(f'\n\n{self.d_i} * {dX}\n\n')
        self.d_i *= dX
        #(np.maximum(a*dX, dX) / dX)
        return self.d_i

class Linear():
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = inputs
        return self.outputs

    def backward(self, dvalues):
        self.d_i = dvalues.copy()
        return self.d_i

# Loss functions
class MSE():
    def forward(self, ypred, yvalue):
        sample_loss = np.mean((yvalue - ypred) ** 2, axis=-1)
        return sample_loss

    def backward(self, dvalues, yvalue):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.d_i = -2 * (yvalue - dvalues) / outputs
        self.d_i = self.d_i / samples
        return self.d_i

# Optimizers
class SGD:
    def __init__(self, alpha=0.001):
        self.a = alpha

    def optimize(self, layer):
        # input(f'optimize with {layer.d_w}')
        layer.weights += -self.a * layer.d_w
        layer.biases += -self.a * layer.d_b




# Network Components
class Layer:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)) #np.random.randn(1, n_neurons)
        self.activation = LReLU()

    def forward(self, inputs: int):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        return self.activation.forward(self.outputs)

    def backward(self, dvalues: list):
        layer_dval = self.activation.backward(dvalues)
        self.d_w = np.dot(self.inputs.T, layer_dval)
        self.d_b = np.sum(layer_dval, axis=0, keepdims=True)
        self.d_i = np.dot(layer_dval, self.weights.T)
        # input(f'd inputs: {self.d_i}')
        return self.d_i


class Output(Layer):
    def __init__(self, n_inputs, n_neurons):
        super().__init__(n_inputs, n_neurons)
        self.activation = Linear()


class Network:
    def __init__(self, n_inputs, n_outputs): # working
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.layers = list()
        self.out_layer = Output(n_inputs, n_outputs)
        self.loss_func = MSE()
        self.optimizer = SGD()

    def add_layer(self, n_neu): # working
        # if layers is empty, n_inputs is based from the input data
        # else n_inputs will be based on the number of neurons of the last layer
        if not self.layers:
            self.layers += [Layer(self.n_inputs, n_neu)]
        else:
            self.layers += [Layer(self.layers[-1].weights.shape[1], n_neu)]
        self.out_layer = Output(self.layers[-1].weights.shape[1], self.n_outputs)
        pass

    def print_layers(self):
        '''
        print the dimension of each hidden layer and the output layer
        '''
        for layer in self.layers:
            print(layer.weights.shape)
        print(f'\noutput layer: {self.out_layer.weights.shape}\n')
        pass

    def forward_prop(self, inputs):
        self.batch = inputs
        for layer in self.layers:
            self.batch = layer.forward(self.batch)
        self.batch = self.out_layer.forward(self.batch)
        return self.batch

    def gradient_descent(self, data):
        # Backward pass
        self.loss_func.forward(self.batch, data)
        dinputs = self.loss_func.backward(self.batch, data)
        dinputs = self.out_layer.backward(dinputs)
        self.layers.reverse()
        for layer in self.layers:
            dinputs = layer.backward(dinputs)
        self.layers.reverse()

        # Gradient descent
        for layer in self.layers:
            self.optimizer.optimize(layer)
        self.optimizer.optimize(self.out_layer)