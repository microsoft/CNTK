from abc import ABCMeta, abstractmethod
import os
import subprocess
import numpy as np
import shutil as sh

from cntk.graph import ComputationNode


_FLOATX = 'float32'
if "CNTK_EXECUTABLE_PATH" not in os.environ:
    raise ValueError(
        "you need to point environmental variable 'CNTK_EXECUTABLE_PATH' to the CNTK binary")

CNTK_EXECUTABLE_PATH = os.environ['CNTK_EXECUTABLE_PATH']
CNTK_TRAIN_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "templates", "cntk_train_template.cntk")
CNTK_TEST_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "templates", "cntk_test_template.cntk")
CNTK_PREDICT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "templates", "cntk_predict_template.cntk")
CNTK_EVAL_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "templates", "cntk_eval_template.cntk")
CNTK_TRAIN_CONFIG_FILENAME = "train.cntk"
CNTK_TEST_CONFIG_FILENAME = "test.cntk"
CNTK_PREDICT_CONFIG_FILENAME = "predict.cntk"
CNTK_EVAL_CONFIG_FILENAME = "eval.cntk"
CNTK_OUTPUT_FILENAME = "out.txt"

# TODO: add validate method
# TODO: overload action methods to support numpy matrices as inputs
# TODO: overload action methods to have versions that do not need reader
# or numpy inputs

_CONTEXT = {}


def get_context(handle='default'):
    # TODO: we need more sanity in the model handling here
    if handle not in _CONTEXT:
        _CONTEXT[handle] = Context(handle)

    return _CONTEXT[handle]


def get_new_context():
    while True:
        new_handle = str(np.random.random())[2:]
        if new_handle not in _CONTEXT:
            return get_context(new_handle)


class AbstractContext(object, metaclass=ABCMeta):
    '''
    This is the abstract CNTK context. It provides an API to run CNTK actions.
    '''

    def __init__(self, name,
                 graph=None,
                 optimizer=None,
                 device_id=-1,
                 root_node=None,
                 clean_up=True):
        '''
        AbstractContext Constructer

        :param name: context name
        :param graph: the computational graph to be used for training, testing and prediction
        :param optimizer: the SGD optimizer to use for training
        :param device_id: whether to use CPU or a specific GPU. -1 for CPU larger values
        :param root_node: the top node of the graph
        :param clean_up: whether the temporary directory should be removed when the context is left
        are the GPUs indices.        

        '''
        if isinstance(name, str):
            tmpdir = name
        else:
            tmpdir = id(name)

        self.directory = os.path.abspath('_cntk_%s' % tmpdir)

        if os.path.exists(self.directory):
            print("Directory '%s' already exists" %
                  self.directory)
        else:
            os.mkdir(self.directory)

        self.name = name
        self.optimizer = optimizer
        self.device_id = device_id
        self.clean_up = clean_up
        self.input_nodes = set()
        self.root_node = root_node

    def __enter__(self):
        _CONTEXT[self.name] = self

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        del _CONTEXT[self.name]

        if self.clean_up:
            import shutil
            shutil.rmtree(self.directory)

    def to_description(self):
        '''
        Generating the CNTK configuration for the root node.
        '''
        return self.root_node.to_description()

    def _generate_train_config(self, reader, override_existing):
        '''
        Generates the configuration file for the train action.
        :param reader: the reader to use for reading the data
        :param override_existing: if the folder exists already override it
        '''
        
        model_dir = os.path.join(self.directory, 'Models')
        if os.path.exists(model_dir) and os.listdir(model_dir) == []:
            if override_existing:                
                print ("Overriding the existing models")
                sh.rmtree(model_dir)
            else:
                raise Exception("Directory '%s' already exists, set the flag override_existing to true if you want to override it"
                    %self.directory)        
        
        tmpl = open(CNTK_TRAIN_TEMPLATE_PATH, "r").read()
        model_filename = os.path.join(model_dir, self.name)
        description, has_inputs, readers = self.to_description()
        if reader:
            readers.append(reader)

        tmpl_dict = {
            'DevideId': self.device_id,
            'ModelDescription': description,
            'ModelPath': model_filename,
            'Reader': '\n'.join(r.generate_config() for r in readers),
            'SGD': self.optimizer.generate_config(),
        }
        return tmpl % tmpl_dict

    def _generate_test_config(self, reader):
        '''
        Generates the configuration file for the test action.
        '''        
        tmpl = open(CNTK_TEST_TEMPLATE_PATH, "r").read()
        model_filename = os.path.join(self.directory, 'Models', self.name)
        tmpl_dict = {
            'DevideId': self.device_id,
            'ModelPath': model_filename,
            'Reader': reader.generate_config(),
        }
        return tmpl % tmpl_dict

    def _generate_predict_config(self, reader):
        '''
        Generates the configuration file for the write action.
        It uses the context's trained model.
        '''
        tmpl = open(CNTK_PREDICT_TEMPLATE_PATH, "r").read()
        model_filename = os.path.join(self.directory, 'Models', self.name)
        output_filename_base = os.path.join(self.directory, 'Outputs', self.name)
        tmpl_dict = {
            'DevideId': self.device_id,
            'ModelPath': model_filename,
            'PredictOutputFile': output_filename_base,
            'Reader': reader.generate_config(),
        }
        return tmpl % tmpl_dict
    
    def _generate_eval_config(self, root_node, reader):
        '''
        Generates the configuration file for write action.
        :param root_node: the node to evaluate. 
        :param reader: the reader used to load the data, None if the network does not have input
        '''        
        model_description, has_input, readers = root_node.to_description()
        if reader:
            readers.append(reader)

        if not has_input and not readers:
            # add dummy input to keep CNTK happy
            # TODO relieve this requirement on CNTK side
            data = [[1, 2], [3, 4]]
            fn = os.path.join(self.directory, 'dummy_input.txt')
            from .reader import NumPyReader
            reader = NumPyReader(data, fn)
            from .ops import Input
            dummy_input_node = Input(2, var_name='dummy_node')
            reader.add_input(dummy_input_node, 0, 2)                        
            model_description += "\ndummy_node=Input(2, tag='output')"
            readers.append(reader)

        tmpl = open(CNTK_EVAL_TEMPLATE_PATH, "r").read()
        output_filename = os.path.join(self.directory, CNTK_OUTPUT_FILENAME)
        tmpl_dict = {
            'DevideId': self.device_id,            
            'OutputFile': output_filename,
            'ModelDescription': model_description,
            'Reader': '\n'.join(r.generate_config() for r in readers),
        }
        return tmpl % tmpl_dict

    @abstractmethod
    def train(self, input_map):
        '''
        Abstract method for the action train.
        :param input_map: mapping of input node to (reader, (start_dim, num_dim))
        '''
        pass

    @abstractmethod
    def test(self, reader):
        '''
        Abstract method for the action test.
        :param reader: the reader to use for this action.
        '''
        pass

    @abstractmethod
    def predict(self, reader):
        '''
        Abstract method for the action write. It evaluated the trained model on 
        the data provided by the reader.
        :param reader: the reader to use for this action.
        '''
        pass

    @abstractmethod
    def eval(self, node, reader=None):
        '''
        Abstract method for the action write. It evaluated the passed node on the
        data provided by the reader.
        :param node: the node to evaluate.
        :param reader: the reader to use for this action.
        '''
        pass


class Context(AbstractContext):

    '''
	This is a sub-class of AbstractContext, use it to run CNTK locally.
    '''

    def _call_cntk(self, config_file_name, config_content):
        '''
        Calls the CNTK exe
        :param config_file_name: the name of the configuration file
        :param config_content: a string containing the configuration
        '''
        filename = os.path.join(self.directory, config_file_name)
        with open(os.path.join(self.directory, filename), "w") as out:
            out.write(config_content)

        subprocess.check_call(
            [CNTK_EXECUTABLE_PATH, "configFile=%s" % filename])

    def train(self, reader, override_existing = True):
        '''
        Run the train action locally.
        :param input_map: mapping of input node to (reader, (start_dim, num_dim))
        :param override_existing: if the folder exists already override it
        '''
        config_content = self._generate_train_config(reader, override_existing)
        self._call_cntk(CNTK_TRAIN_CONFIG_FILENAME, config_content)

    def test(self, reader):
        '''
        Run the test action locally.
        :param reader: the reader used to provide the testing data.
        '''
        config_content = self._generate_test_config(reader)
        self._call_cntk(CNTK_TEST_CONFIG_FILENAME, config_content)

    def predict(self, reader):
        '''
        Run the write action locally, use the trained model of this context.
        :param reader: the reader used to provide the evaluation data.
        '''
        config_content = self._generate_predict_config(reader)
        self._call_cntk(CNTK_PREDICT_CONFIG_FILENAME, config_content)

    def eval(self, node, reader=None):
        '''
        Run the write action locally to evaluate the passed node.
        :param input_map: mapping of input node to (reader, (start_dim, num_dim))
        :param node: the node to evaluate.
        '''
        # FIXME manually setting the tag to output might have side-effects
        if not isinstance(node, ComputationNode):
            raise ValueError('node is not of type ComputationNode, but %s'%type(node))
        node.tag = 'output'
        config_content = self._generate_eval_config(node, reader)
        self._call_cntk(CNTK_EVAL_CONFIG_FILENAME, config_content)

        import glob
        out_file_wildcard = os.path.join(
            self.directory, CNTK_OUTPUT_FILENAME + '.*')
        out_filenames = glob.glob(out_file_wildcard)

        out_filenames = [
            f for f in out_filenames if not f.endswith('out.txt.dummy_node')]

        if len(out_filenames) != 1:
            raise ValueError('expected exactly one file starting with "%s", but got %s' % (
                CNTK_OUTPUT_FILENAME, out_filenames))

        data = np.loadtxt(out_filenames[0])

        return data


class ClusterContext(AbstractContext):
    '''
	This is a sub-class of AbstractContext, use it to submit your workloads to the cluster.
    '''
    pass
