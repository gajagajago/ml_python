import numpy as np

class LogicGate:
    def __init__(self, name, x_data, t_data):
        self.name = name
        self.__x_data = x_data.reshape(4,2)
        self.__t_data = t_data.reshape(4,1)
        
        self.__W = np.random.rand(2,1)
        self.__b = np.random.rand(1,)
        
        self.__learning_rate = 1e-2
        
    def __loss_func(self):
        delta = 1e-7
        z = np.dot(self.__x_data, self.__W) + self.__b
        y = sigmoid(z)
        
        return -np.sum( np.log(y+delta)*self.__t_data + np.log(1-y+delta)*(1-self.__t_data) )
    
    def error_val(self):
        return self.__loss_func()
    
    def train(self):
        f = lambda x : self.__loss_func()
        
        for step in range(10001):
            self.__W -= self.__learning_rate * multi_var_numerical_derivative(f, self.__W)
            self.__b -= self.__learning_rate * multi_var_numerical_derivative(f, self.__b)
            
            if (step % 400 == 0):
                print("STEP %d: W = " % step, self.__W, " b = ", self.__b, " error = ", self.error_val())  
                
    def predict(self, x): # x.shape = (1,2)
        z = np.dot(x, self.__W) + self.__b
        y = sigmoid(z)
        
        return y, 1 if y > 0.5 else 0 
