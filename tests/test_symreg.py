#import sys
#sys.path.insert(1, '../')
import Sym_Reg.SymReg as SymReg
import numpy as np 



def test_model_create():
    model = SymReg.create_model(10,1000,10) 
    assert model != None

def test_model_train_gen_eq():
    model = SymReg.create_model(10,100,10)
    x = np.random.randint(15, size=(1, 5))
    y = np.random.randint(15, size=(1,1))
    history = model.fit(x,y)
    best_eq_str = SymReg.gen_equation(model)
    assert history and best_eq_str != None

