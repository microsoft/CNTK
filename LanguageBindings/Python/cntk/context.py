from abc import ABCMeta, abstractmethod
import os
import subprocess
import numpy as np


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

_CONTEXT = {}
def get_context(model_or_path=None):
    # TODO: we need more sanity in the model handling here
    if model_or_path is None:
        model_or_path = 'default'

    if model_or_path not in _CONTEXT:
        _CONTEXT[model_or_path] = Context(model_or_path)

    return _CONTEXT[model_or_path] 

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
        
        self.name = name
        self.macros = []  
        self.graph = graph or Graph()
        self.optimizer = optimizer
        self.device_id = device_id
        self.clean_up = clean_up
        
    def __enter__(self):
        _CONTEXT[self.name] = self

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        del _CONTEXT[self.name]

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
        model_filename = os.path.join(self.directory, 'Models', self.name)
        tmpl_dict = {
                'DevideId':self.device_id,
                'ModelDescription':self.graph.to_graph_description(),
                'ModelModelPath': model_filename,
                'Reader':reader_config,
                'SGD':self.optimizer.generate_config(),
                } 
        return tmpl%tmpl_dict
        
    def _generate_test_config(self, reader):
        """Generates the configuration file for the test action.
        """                
        raise NotImplementedError        
    
    def _generate_predict_config(self, reader):
        """Generates the configuration file for the write action.
        It uses the context's trained model.
        """                
        raise NotImplementedError 
    
    def _generate_eval_config(self, root_node, input_map):
        """Generates the configuration file for write action.
        :param root_node: the node to evaluate. 
        :param input_map: map from feature node to reader, dimensions
        """                
        not_assigned = []

        for node in self.graph.feature_nodes:
            if node not in input_map:
                not_assigned.append(node)

        if not_assigned:
            raise ValueError('Cannot create the configuration, because the ' +
            'following input nodes are missing corresponding input readers: ' +
            ", ".join(not_assigned))

        # TODO to corresponding tests for label_nodes (e.g. do we have a label
        # mapping per label node, etc.)

        # TODO factor out reader config output so that train/test can use it
        model_description = root_node.to_graph_description()

        if not self.graph.feature_nodes:
            #import ipdb;ipdb.set_trace()
            # add dummy input to keep CNTK happy 
            # TODO relieve this requirement
            
            data = [[1,2], [3,4]]
            fn = os.path.join(self.directory, 'dummy_input.txt')
            from .reader import NumPyReader
            reader = NumPyReader(data, fn)
            from .ops import Input
            dummy_input_node = Input(2, ctx=self)
            dummy_input_node.var_name='dummy_node'
            input_map = {dummy_input_node:(reader, (0,2))}
            model_description += "\ndummy_node=Input(2, 1, tag='feature')"

        tmpl = open(CNTK_EVAL_TEMPLATE_PATH, "r").read()
        readers = set() 
        node_dimensions = []
        for node, (reader, dims) in input_map.items():
            if not node.var_name:
                raise ValueError("Node '%s' does not have a variable name assigned yet."%str(node))

            start, num_dims = dims
            node_dimensions.append('''\
%s = [
               start=%i
               dim=%i
           ]'''%(node.var_name, start, num_dims))
                        

            readers.add(reader)

        node_dimensions = "\n".join(node_dimensions)

        # make sure every reader is configured only once
        reader_configs = []
        for reader in readers:
            reader_configs.append(reader.generate_config())
        reader_configs = "\n".join(reader_configs)

        output_filename = os.path.join(self.directory, CNTK_OUTPUT_FILENAME)
        tmpl_dict = {
                'DevideId':self.device_id,
                'NodeDimensions':node_dimensions,
                'Reader':reader_configs,
                'OutputFile':output_filename,
                'ModelDescription':model_description
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
    
    def eval(self, node, input_map):
        """Run the write action locally to evaluate the passed node.
        :param input_map: mapping of input node to (reader, (start_dim, num_dim))
        :param node: the node to evaluate.
        """            
        config_content = self._generate_eval_config(node, input_map) 
        self._call_cntk(CNTK_EVAL_CONFIG_FILENAME, config_content) 

        import glob
        out_file_wildcard = os.path.join(self.directory, CNTK_OUTPUT_FILENAME+'.*')
        out_filenames = glob.glob(out_file_wildcard)
        if len(out_filenames) != 1:
            raise ValueError('expected exactly one file starting with "%s", but got %s'%(CNTK_OUTPUT_FILENAME, out_filenames))

        data = np.loadtxt(out_filenames[0])

        return data

class ClusterContext(AbstractContext):
    """This is a sub-class of AbstractContext, use it to submit your workloads to the cluster.
    """    
    pass

from .graph import Graph
