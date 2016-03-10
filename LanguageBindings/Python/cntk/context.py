from abc import ABCMeta, abstractmethod
import os
import subprocess

_FLOATX = 'float32'
if "CNTK_EXECUTABLE_PATH" not in os.environ:
    raise ValueError("you need to point environmental variable 'CNTK_EXECUTABLE_PATH' to the CNTK binary")

CNTK_EXECUTABLE_PATH = os.environ['CNTK_EXECUTABLE_PATH']
CNTK_TRAIN_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "cntk_train_template.cntk")
CNTK_TEST_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "cntk_test_template.cntk")
CNTK_PREDICT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "cntk_predict_template.cntk")
CNTK_EVAL_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "cntk_eval_template.cntk")
CNTK_TRAIN_CONFIG_FILENAME = "train.cntk"
CNTK_TEST_CONFIG_FILENAME = "test.cntk"
CNTK_PREDICT_CONFIG_FILENAME = "predict.cntk"
CNTK_EVAL_CONFIG_FILENAME = "eval.cntk"
CNTK_OUTPUT_FILENAME="out.txt"

'''This is the abstract CNTK context. It provides an API to run CNTK actions
'''
class AbstractContext(object, metaclass=ABCMeta):

    '''AbstractContext Constructer
    :param name: context name
    :param graph: the computational graph to be used for training, testing and prediction
    :param optimizer: the SGD optimizer to use for training
    :param device_id: whether to use CPU or a specific GPU. -1 for CPU larger values
    are the GPUs indices.
    '''
	#TODO: add validate method
    #TODO: overload action methods to support numpy matrices as inputs
    def __init__(self, name, graph = None, optimizer = None, device_id = -1):
        self.directory = os.path.abspath('_cntk_%s'%id(name))
        if os.path.exists(self.directory):
            print("Directory '%s' already exists - overwriting data."%self.directory) 
        else:
            os.mkdir(self.directory)
        
        self.macros = []  
        self.graph = graph
        self.optimizer = optimizer
        self.device_id = device_id
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass    
    
    '''Add a macro file to be referenced from all configurations of this context.
    :param path: path of the macro file.    
    '''    
    def add_macro(self, path):        
        self.macros.append(path)    

    '''Generates the configuration file for the train action.
    '''        
    def _generate_train_config(self):
        raise NotImplementedError        
    
    '''Generates the configuration file for the test action.
    '''        
    def _generate_test_config(self):
        raise NotImplementedError        
    
    '''Generates the configuration file for the write action.
    It uses the context's trained model.
    '''        
    def _generate_predict_config(self):
        raise NotImplementedError 
    
    '''Generates the configuration file for write action.
    :param node: the node to evaluate. 
    '''        
    def _generate_eval_config(self, node):
        raise NotImplementedError        
        
    '''Abstract method for the action train.
    :param reader: the reader to use for this action.
    '''
    @abstractmethod
    def train(self, reader):
        pass 
    
    '''Abstract method for the action test.
    :param reader: the reader to use for this action.
    '''
    @abstractmethod
    def test(self, reader):
        pass     
    
    '''Abstract method for the action write. It evaluated the trained model on 
    the data provided by the reader.
    :param reader: the reader to use for this action.
    '''
    @abstractmethod
    def predict(self, reader):
        pass     
    
    '''Abstract method for the action write. It evaluated the passed node on the
    data provided by the reader.
    :param node: the node to evaluate.
    :param reader: the reader to use for this action.
    '''    
    @abstractmethod
    def eval(self, node, reader):
        pass 
    
'''This is a sub-class of AbstractContext, use it to run CNTK locally.
'''
class Context(AbstractContext):    
    '''Calls the CNTK exe
    :param config_file_name: the name of the configuration file
    :param config_content: a string containing the configuration
    '''
    def _call_cntk(self, config_file_name, config_content):
        filename = os.path.join(self.directory, config_file_name)        
        with open(os.path.join(self.context.directory, filename), "w") as out:
            out.write(config_content)            
        subprocess.check_call([CNTK_EXECUTABLE_PATH, "configFile=%s"%filename])        
    
    
    '''Run the train action locally.
    :param reader: the reader used to provide the training data.
    '''
    def train(self, reader):
        config_content = self._generate_train_config()         
        self._call_cntk(CNTK_TRAIN_CONFIG_FILENAME, config_content)        
    
    '''Run the test action locally.
    :param reader: the reader used to provide the testing data.
    '''    
    def test(self, reader):
        config_content = self._generate_test_config() 
        self._call_cntk(CNTK_TEST_CONFIG_FILENAME, config_content) 

    '''Run the write action locally, use the trained model of this context.
    :param reader: the reader used to provide the prediction data.
    '''    
    def predict(self, reader):
        config_content = self._generate_predict_config() 
        self._call_cntk(CNTK_PREDICT_CONFIG_FILENAME, config_content) 
    
    '''Run the write action locally to evaluate the passed node.
    :param reader: the reader used to provide the prediction data.
    :param node: the node to evaluate.
    '''    
    def eval(self, node, reader):
        config_content = self._generate_eval_config() 
        self._call_cntk(CNTK_EVAL_CONFIG_FILENAME, config_content) 

'''This is a sub-class of AbstractContext, use it to submit your workloads to the cluster.
'''
class ClusterContext(AbstractContext):
    pass