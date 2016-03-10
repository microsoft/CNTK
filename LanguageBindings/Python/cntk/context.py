from abc import ABCMeta, abstractmethod
import os
import subprocess

_FLOATX = 'float32'
if "CNTK_EXECUTABLE_PATH" not in os.environ:
    raise ValueError("you need to point environmental variable 'CNTK_EXECUTABLE_PATH' to the CNTK binary")

CNTK_EXECUTABLE_PATH = os.environ['CNTK_EXECUTABLE_PATH']
CNTK_TRAIN_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "cntk_train_template.cntk")
CNTK_TEST_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "cntk_test_template.cntk")
CNTK_PREDICT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "cntk_predict_template.cntk")
CNTK_EVAL_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "cntk_eval_template.cntk")
CNTK_TRAIN_CONFIG_FILENAME = "train.cntk"
CNTK_TEST_CONFIG_FILENAME = "test.cntk"
CNTK_PREDICT_CONFIG_FILENAME = "predict.cntk"
CNTK_EVAL_CONFIG_FILENAME = "eval.cntk"
CNTK_OUTPUT_FILENAME="out.txt"

#TODO: add validate method
#TODO: overload action methods to support numpy matrices as inputs
#TODO: overload action methods to have versions that do not need reader or numpy inputs

class AbstractContext(object, metaclass=ABCMeta):
    """This is the abstract CNTK context. It provides an API to run CNTK actions
    """
    def __init__(self, name, graph = None, optimizer = None, device_id = -1,
            clean_up=True):
        """AbstractContext Constructer
        
        :param name: context name
        :param graph: the computational graph to be used for training, testing and prediction
        :param optimizer: the SGD optimizer to use for training
        :param device_id: whether to use CPU or a specific GPU. -1 for CPU larger values
        :param clean_up: whether the temporary directory should be removed when the context is left
        are the GPUs indices.
        
        """
        self.directory = os.path.abspath('_cntk_%s'%id(name))
        if os.path.exists(self.directory):
            print("Directory '%s' already exists - overwriting data."%self.directory) 
        else:
            os.mkdir(self.directory)
        
        self.macros = []  
        self.graph = graph
        self.optimizer = optimizer
        self.device_id = device_id
        self.clean_up = clean_up
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.clean_up:
            import shutil
            shutil.rmtree(self.directory)
    
    def add_macro(self, path):        
        """Add a macro file to be referenced from all configurations of this context.
        :param path: path of the macro file.    
        """            
        self.macros.append(path)    

    def _generate_train_config(self, reader):
        """Generates the configuration file for the train action.
        """                
        tmpl = open(CNTK_TRAIN_TEMPLATE_PATH, "r").read()
        reader_config = reader.generate_config()
        output_filename = os.path.join(self.context.directory, CNTK_OUTPUT_FILENAME)
        return tmpl%{'modelDescription': self.graph.to_description(), 'reader':reader_config, 'outputFile':output_filename}
    
    def _generate_test_config(self, reader):
        """Generates the configuration file for the test action.
        """                
        raise NotImplementedError        
    
    def _generate_predict_config(self, reader):
        """Generates the configuration file for the write action.
        It uses the context's trained model.
        """                
        raise NotImplementedError 
    
    def _generate_eval_config(self, node, reader):
        """Generates the configuration file for write action.
        :param node: the node to evaluate. 
        """                
        tmpl = open(CNTK_EVAL_TEMPLATE_PATH, "r").read()
        reader_config = reader.generate_config()
        output_filename = os.path.join(self.directory, CNTK_OUTPUT_FILENAME)
        tmpl_dict = {
                'Reader':reader_config,
                'OutputFile':output_filename,
                'ModelDescription':node.to_description()
                } 
        return tmpl%tmpl_dict
                
    @abstractmethod
    def train(self, reader):
        """Abstract method for the action train.
        :param reader: the reader to use for this action.
        """        
        pass 
    
    @abstractmethod
    def test(self, reader):
        """Abstract method for the action test.
        :param reader: the reader to use for this action.
        """        
        pass     
    
    @abstractmethod
    def predict(self, reader):
        """Abstract method for the action write. It evaluated the trained model on 
        the data provided by the reader.
        :param reader: the reader to use for this action.
        """        
        pass     
    
    @abstractmethod
    def eval(self, node, reader):
        """Abstract method for the action write. It evaluated the passed node on the
        data provided by the reader.
        :param node: the node to evaluate.
        :param reader: the reader to use for this action.
        """            
        pass 
    
class Context(AbstractContext):    
    """This is a sub-class of AbstractContext, use it to run CNTK locally.
    """
    
    def _call_cntk(self, config_file_name, config_content):
        """Calls the CNTK exe
        :param config_file_name: the name of the configuration file
        :param config_content: a string containing the configuration
        """
        filename = os.path.join(self.directory, config_file_name)        
        with open(os.path.join(self.directory, filename), "w") as out:
            out.write(config_content)            
        subprocess.check_call([CNTK_EXECUTABLE_PATH, "configFile=%s"%filename])        
    
    def train(self, reader):
        """Run the train action locally.
        :param reader: the reader used to provide the training data.
        """        
        config_content = self._generate_train_config(reader)         
        self._call_cntk(CNTK_TRAIN_CONFIG_FILENAME, config_content)        
    
    def test(self, reader):
        """Run the test action locally.
        :param reader: the reader used to provide the testing data.
        """            
        config_content = self._generate_test_config(reader) 
        self._call_cntk(CNTK_TEST_CONFIG_FILENAME, config_content) 

    def predict(self, reader):
        """Run the write action locally, use the trained model of this context.
        :param reader: the reader used to provide the prediction data.
        """            
        config_content = self._generate_predict_config(reader) 
        self._call_cntk(CNTK_PREDICT_CONFIG_FILENAME, config_content) 
    
    def eval(self, node, reader):
        """Run the write action locally to evaluate the passed node.
        :param reader: the reader used to provide the prediction data.
        :param node: the node to evaluate.
        """            
        config_content = self._generate_eval_config(node, reader) 
        self._call_cntk(CNTK_EVAL_CONFIG_FILENAME, config_content) 

        import glob
        out_file_wildcard = os.path.join(self.directory, CNTK_OUTPUT_FILENAME+'.*')
        out_filenames = glob.glob(out_file_wildcard)
        if len(out_filenames)!=1:
            raise ValueError('expected exactly one file starting with "%s", but got %s'%(CNTK_OUTPUT_FILENAME, out_filenames))

        data = np.loadtxt(out_filenames[0])

        return data

class ClusterContext(AbstractContext):
    """This is a sub-class of AbstractContext, use it to submit your workloads to the cluster.
    """    
    pass
