import numpy as np

class nn:
    def __init__(self, X, Y,lr:float=1.2,units:int=64,num_iterations:int = 10000, print_cost:bool=False):
        self.X = X
        self.Y = Y
        self.n_x = X.shape[0]
        self.n_h = units
        self.n_y = Y.shape[0]
        self.W1 = np.random.randn(self.n_h,self.n_x) * 0.01
        self.b1 = np.zeros((self.n_h, 1)) 
        self.W2 =np.random.randn(self.n_y,self.n_h) * 0.01
        self.b2 = np.zeros((self.n_y, 1)) 
        self.lr = lr
        self.cost = 0
        self.accuracy = 0
        self.iterations = 0
        self.cost_history = []
        self.accuracy_history = []
        self.parameters_history = []
    
    def sigmoid(self,z):
        s = 1 / (1 + np.exp(-z))    
        return s

    def forward_propagation(self):

        Z1 = np.dot(self.W1, self.X) + self.b1     # Z1 = W1*X + b1
        A1 = np.tanh(Z1)                           # A1 = tanh(Z1)
        Z2 = np.dot(self.W2,A1) + self.b2          # Z2 = W2*X + b2
        A2 = self.sigmoid(Z2)                      # A2 = sigmoid(Z2)

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}
    
        return A2, cache 

    def compute_cost(self,A2):
        """
        Compute the cross-entropy cost
        """
        m = self.Y.shape[1] 
        logprobs = np.multiply(np.log(A2), self.Y) + np.multiply((1 - self.Y), np.log(1 - A2)) 
        cost = - np.sum(logprobs) / m
        # cost = np.array((- 1 / m) * np.sum(Y * np.log(A2) + (1 - Y) * (np.log(1 - A2))))
        cost = float(np.squeeze(cost)) 
        return cost

    def backward_propagation(self, cache):

        m = self.X.shape[1]

        A1 = cache['A1']
        A2 = cache['A2']
        
        dZ2 = A2 - self.Y
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis = 1, keepdims=True)
        dZ1 = np.multiply(np.dot(self.W2.T, dZ2), 1 - np.power(A1, 2))
        dW1 = (1/m ) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis = 1, keepdims=True)
        
        grads = {"dW1": dW1,
                "db1": db1,
                "dW2": dW2,
                "db2": db2}
        
        return grads

    def update_parameters(self, grads):
        
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']
        
        self.W1 = self.W1 - self.lr * dW1
        self.b1 = self.b1 - self.lr * db1
        self.W2 = self.W2 - self.lr * dW2
        self.b2 = self.b2 - self.lr * db2
        
        parameters = {"W1": self.W1,
                    "b1": self.b1,
                    "W2": self.W2,
                    "b2": self.b2}
        
        return parameters

    def train(self,num_iterations:int = 10000, print_cost:bool=True):
        
        for i in range(0, num_iterations):
            
            A2, cache = self.forward_propagation()
            
            cost = self.compute_cost(A2)
            self.cost_history.append(cost)

            grads = self.backward_propagation(cache)
    
            parameters = self.update_parameters(grads)
            
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

        return parameters

model = nn(X,Y,units=4)
model.train()
