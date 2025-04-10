import os 
os.environ["JULIA_NUM_THREADS"] = "8"
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
from Sym_Reg.SymReg import create_model
from Sym_Reg.SymReg import gen_equation

data = pd.read_csv("Org-Data/OriginalData.csv")  # Assume dataset has 12 input features and 2 labels

X = data.iloc[:, :-2].values  # First 12 columns as inputs
y_1 = data.iloc[:, -2:-1].values #RT
y_2 =data.iloc[:, -1:].values #HT

X_train, X_test, y_train_1, y_test_1 = train_test_split(X, y_1, test_size=0.5)


'''
model = pysr.PySRRegressor(
    precision=32,
    niterations=300,
    binary_operators=["+", "*", "-", "/", "^"],
    unary_operators=["sin", "exp", "log","sqrt"],
    population_size=2500,
    maxsize = 100,
    parsimony=1e-6,
)

'''

model = create_model(10,1000,10)
model.fit(X_train, y_train_1)

best_eq_str = gen_equation(model)

#best_eq = model.get_best()
#best_eq_str = best_eq.equation  # This is the full symbolic equation as a string

with open("best_equation.txt", "w") as f:
    f.write(best_eq_str)


