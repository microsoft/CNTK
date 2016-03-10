
'''This is the Stochastic Gradien Descent optimizer used to train the networks
'''
class SGD(dict):        
    def __init__(self, epoch_size, minibatch_size, learning_ratesPerMB, learning_rates_per_sample, momentum_per_mb, max_epochs, dropout_rate):
        ''' SGD constructor
        :param epoch_size: the number of samples to use in each epoch. An intermediate
        model and other check point information are saved for each epoch. When set 
        to 0 the whole dataset size is used. 
        :param minibatch_size: the minibatch size. The default value is 256. You 
        can use different values for different epochs, e.g., 128*2:1024 means using
        minibatch size of 128 for the first two epochs and then 1024 for the rest. 
        :param learning_ratesPerMB: the learning rates per epoch. You can use different
        values for different epochs, e.g., 0.8*10:0.2 means use the learning rate 0.8
        for the first 10 epochs and then 0.2 for the rest. 
        :param learning_rates_per_sample: the learning rates per sample per epoch.
        If you want your learning rate to vary in function of the minibatch size,
        you can use  learningRatesPerSample  instead of  learningRatesPerMB . This
        will automatically increase the learning rate for the minibatch when the
        minibatch size increases, that is, the value defined here will be multiplied
        by the minibatch size to get the learning rate. You can use different values
        for different epochs, e.g., 0.008*10:0.002 means use the learning rate 0.008
        for the first 10 epochs and then 0.002 for the rest. 
        :param momentum_per_mb: The default value is 0.9. Different values can
        be given to different epochs. It is important to note that CNTK has a particular
        behaviour when dealing with momentum, the learning rate is automatically further
        scaled by a factor of (1 – momentum).
        :param momentum_per_sample: similarly to learning rate, momentum can be defined
        on the sample level, also, different values can be given to different epochs.
        :param max_epochs: the maximum number of epochs to run. 
        :param dropout_rate: the dropout rate per epoch. The default value is 0.
        '''    
            
        pass
