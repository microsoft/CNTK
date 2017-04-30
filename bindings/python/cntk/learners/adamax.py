from cntk.learners import  UserLearner  
import numpy as np
import math

class adamax(UserLearner):
    def __init__(self, parameters, lr_schedule):
        super(adamax33, self).__init__(parameters, lr_schedule)
        self.count = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-2
        self.m = []
        self.n = []
       
       
    def update(self, gradient_values, training_sample_count, sweep_end):
        eta = self.learning_rate()

        if(self.count==0):
            for p, g in gradient_values.items():
                P = p.value
                self.m.append((np.zeros(P.shape)))
                self.n.append((np.zeros(P.shape)))
               
        self.count = self.count+1
        currentlayer=0
       
        for p, g in gradient_values.items():         
            p.value = p.value-(eta/(1-math.pow(self.beta1, self.count)))*self.m[currentlayer]/(self.n[currentlayer]+self.epsilon)
            self.m[currentlayer] = (1-self.beta1)*g.to_ndarray()+self.beta1*self.m[currentlayer]
            self.n[currentlayer] = np.maximum(self.beta2*self.n[currentlayer], np.abs(g.to_ndarray()))
            currentlayer = currentlayer + 1
        return True
