import os
from abc import ABCMeta, abstractmethod

_FLOATX = 'float32'
if "CNTK_EXECUTABLE_PATH" not in os.environ:
    raise ValueError("you need to point environmental variable 'CNTK_EXECUTABLE_PATH' to the CNTK binary")

CNTK_EXECUTABLE_PATH = os.environ['CNTK_EXECUTABLE_PATH']
CNTK_TRAIN_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "cntk_train_template.cntk")
CNTK_PREDICT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "cntk_predict_template.cntk")
CNTK_TRAIN_CONFIG_FILENAME = "train.cntk"
CNTK_PREDICT_CONFIG_FILENAME = "predict.cntk"
CNTK_OUTPUT_FILENAME="out.txt"

'''
'''
class AbstractContext(object, metaclass=ABCMeta):

    '''
    '''
    def __init__(self, name, optimizer = None, device_id = -1):
        self.directory = os.path.abspath('_cntk_%s'%id(name))
        if os.path.exists(self.directory):
            print("Directory '%s' already exists - overwriting data."%self.directory) 
        else:
            os.mkdir(self.directory)
        
        self.macros = []        
        self.optimizer = optimizer
        self.device_id = device_id
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass    
    
    '''
    '''    
    def add_macro(self, path):        
        self.macros.append(path)    

    '''
    '''        
    def _generate_train_config(self):
        pass        
    
    '''
    '''        
    def _generate_test_config(self):
        pass        
    
    '''
    '''        
    def _generate_eval_config(self):
        pass        
        
    '''
    '''
    @abstractmethod
    def train(self, reader):
        pass 
    
    '''
    '''
    @abstractmethod
    def test(self, reader):
        pass     
    
    '''
    '''    
    @abstractmethod
    def eval(self, reader):
        pass 
    
'''
'''
class Context(AbstractContext):
    '''
    '''
    def __init__(self, name, optimizer = None, device_id = -1):        
        super(Context, self).__init__(name, optimizer, device_id)
    
    '''
    '''    
    def train(self, reader):
        self._generate_train_config() 
        #TODO: run exe
    
    '''
    '''    
    def test(self, reader):
        self._generate_test_config() 
        #TODO: run exe
    
    '''
    '''    
    def eval(self, reader):
        self._generate_eval_config() 
        #TODO: run exe

'''
'''
class ClusterContext(AbstractContext):
    pass