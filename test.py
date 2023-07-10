import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
from Neural_Net import *
from data_processing import *
print(f'imports complete\n')

# get data
data = np.array(pd.read_csv('Div Data.csv'))
# print(data)

# build neural network
# nn = Network(2, 1)
# nn.add_layer(10)
# nn.add_layer(10)
# nn.print_layers()
l1 = Layer(2, 10)
a1 = LReLU()
l2 = Layer(10, 1)
a2 = Linear()
cf = MSE()
op = 

arr = np.random.randn(5, 5)
arr[arr > 0] = arr[arr > 0] * 100

# batch = data[:, :-1]
# y = data[:, -1]

# Training montage
test = np.array([[10, 5]])
for epoch in range(100):
    # shuffle data set by accessing with shuffled keys
    keys = np.arange(0, 1000)
    np.random.shuffle(keys)
    while keys.shape[0] != 0:
        batch = data[keys[:5], :-1]
        y = data[keys[:5], -1]
        y = y.reshape((y.shape[0], 1))
        keys = keys[5:]
        print(f'batch: {batch} \n\n Y: {y}')
        # print(nn.forward_prop(batch))
        # nn.gradient_descent(y)
    input()

class basta:
    def __init__(self, num):
        self.x = num
    def __str__(self) -> str:
        return str(self.x)

    def add(self, num):
        self.x += num



# a = [[x*y for y in range(11)] for x in range(11)]
# print(np.array(a))

# can i do this now
# okay lets do another one here

# a new branch
