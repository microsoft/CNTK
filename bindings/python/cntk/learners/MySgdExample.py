from cntk import UserLearner

class MySgd(UserLearner):

    def __init__(self, parameters, lr_schedule):
        super(MySgd, self).__init__(parameters, lr_schedule)
        self.v = []
        self.count = 0
        self.alpha = 0.4
        
    def update(self, gradient_values, training_sample_count, sweep_end):
        eta = self.learning_rate() 
    
        layer = 0    
        for p, g in gradient_values.items():
            if self.count == 0:
               self.v.append(-eta * g.to_ndarray())
            else:
                self.v[layer] = self.alpha*self.v[layer] - eta * g.to_ndarray()
            p.value = p.value + self.v[layer]
            layer += 1
            
        self.count += 1     
        return True
