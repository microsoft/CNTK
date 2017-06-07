from cntk.learners import UserLearner
import numpy as np
import math

class nadam(UserLearner):

    '''
    Creates an Nadam learner instance to learn the parameters. 
    '''
    
    def __init__(self, parameters, lr_schedule):
        super(nadam, self).__init__(parameters, lr_schedule)
        self.m = []
        self.n = []
        self.mu = 0.99
        self.mu_pro = 0.99
        self.nu = 0.999
        self.epsilon = 1e-8
        self.t = 0
        
    def update(self, gradient_values, training_sample_count, sweep_end):
        lr = self.learning_rate()
        
        if(self.t==0):
            for p, g in gradient_values.items():
                P = p.value
                self.m.append(np.zeros(P.shape))
                self.n.append(np.zeros(P.shape))
                
        self.mu_pro = self.mu_pro*self.mu*(1-0.5*math.pow(0.96,self.t/250))
        self.t += 1
        idx = 0
        for p, g in gradient_values.items():
            self.m[idx] = self.mu*self.m[idx] + (1-self.mu)*g.to_ndarray()
            self.n[idx] = self.nu*self.n[idx] + (1-self.nu)*g.to_ndarray()*g.to_ndarray()
            g_hat = g.to_ndarray()/(1-self.mu_pro)
            m_hat = self.m[idx]/(1-self.mu_pro*self.mu*(1-0.5*math.pow(0.96,(self.t+1)/250)))
            n_hat = self.n[idx]/(1-math.pow(self.nu,self.t))
            m_bar = (1-self.mu*(1-0.5*math.pow(0.96,self.t/250)))*g_hat + self.mu*(1-0.5*math.pow(0.96,(self.t+1)/250))*m_hat
            p.value = p.value - lr*m_bar/(pow(n_hat,0.5)+self.epsilon)
            
            idx += 1
        return True