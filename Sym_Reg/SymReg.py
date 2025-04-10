import numpy as np
import pandas as pd
import scipy.optimize as opt
import pysr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def create_model(iters,popsize,size):
    model = pysr.PySRRegressor(
    precision=32,
    niterations=iters,
    binary_operators=["+", "*", "-", "/", "^"],
    unary_operators=["sin", "exp", "log","sqrt"],
    population_size=popsize,
    maxsize = size,
    parsimony=1e-6,
)
    return model 

def gen_equation(model):
    best_eq = model.get_best()
    best_eq_str = best_eq.equation
    return best_eq_str