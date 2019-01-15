# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

dataset = pd.read_csv('./train.csv')

X, y = dataset.iloc[:,1:], dataset.iloc[:,:1]

del dataset

class Layer:
    def __init__(self):
        pass
    
    def forward(self, input):
        return input
    
    def backward(self, input, grad):
        m = input.shape[0]
        d = np.eye(m)
        
        return np.dot(grad, d)

class ReLU:
    def __init__(self):
        pass
    
    def forward(self, input):
        return np.maximum(input, np.zeros(input.shape))
    
    def backward(self, input, grad):
        r_grad = input > 0
        return grad * r_grad

class Dense:
    def __init__(self, in_size, out_size, eta=0.1):
        self.w = np.random.randn(in_size, out_size) * 0.01
        self.b = np.zeros(out_size)

    def forward(self, input):
        return np.dot(input, self.w) + self.b
    
    def backward(self, input, grad):
        grad_input = np.dot(grad, self.w.T)
        
        grad_w = np.dot(input.T, grad)
        grad_b = np.sum(grad, axis=0)
        
        self.w = self.w - self.eta * grad_w
        self.b = self.b - self.eta * grad_b
        
        return grad_input

class SoftmaxCrossEntropy:
    def __init__(self):
        pass
    
    def foward(self, logits, y):
        m = logits.shape[0]
        
        l = logits[np.arange(m),y]
        e = -l + np.log(np.sum(np.exp(logits), axis=-1))
        
        return e

    def backward(self, logits, y):
        m = logits.shape[0]
        
        ones = np.zeros_like(logits)
        ones[np.arange(m),y] = 1
        
        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
        
        return (-ones + softmax) / m

class Sequential:
    def __init__(self):
        self.network = []
    
    def add(self, model):
        pass

    def forward(self, input):
        pass
    
    def backward(self, input, grad):
        pass

model = Sequential()

model.add(Dense(X.shape[0], 100))
model.add(ReLU())
model.add(Dense(100, 200))
model.add(ReLU())
model.add(Dense(200,10))








