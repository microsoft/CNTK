from abc import ABCMeta, abstractmethod
import os
import re
import sys
import subprocess
import numpy as np
import shutil as sh

from cntk.graph import ComputationNode
from cntk.ops.cntk1 import NewReshape
from cntk.utils import CNTK_EXECUTABLE_PATH


_FLOATX = 'float32'

CNTK_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
CNTK_TRAIN_TEMPLATE_PATH = os.path.join(
        CNTK_TEMPLATE_DIR, "cntk_train_template.cntk")
CNTK_TEST_TEMPLATE_PATH = os.path.join(
        CNTK_TEMPLATE_DIR, "cntk_test_template.cntk")
CNTK_PREDICT_TEMPLATE_PATH = os.path.join(
        CNTK_TEMPLATE_DIR, "cntk_predict_template.cntk")
CNTK_EVAL_TEMPLATE_PATH = os.path.join(
        CNTK_TEMPLATE_DIR, "cntk_eval_template.cntk")
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
                 device_id=-1,
                 root_node=None,
                 clean_up=True):
        '''
        AbstractContext Constructer

        :param name: context name
        :param graph: the computational graph to be used for training, testing and prediction        
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

    def _generate_train_config(self, optimizer, reader, override_existing):
        '''
        Generates the configuration file for the train action.
        :param optimizer: the SGD optimizer to use for training
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
        description, has_inputs, readers = self.root_node.to_config()        
        if reader:
            readers.append(reader)

        tmpl_dict = {
            'DevideId': self.device_id,
            'ModelDescription': description,
            'ModelPath': model_filename,
            'Reader': '\n'.join(r.generate_config() for r in readers),
            'SGD': optimizer.generate_config(),
        }
        return tmpl % tmpl_dict

    def _generate_test_config(self, reader):
        '''
        Generates the configuration file for the test action.
        '''        
        tmpl = open(CNTK_TEST_TEMPLATE_PATH, "r").read()
        model_filename = os.path.join(self.directory, 'Models', self.name)

        # if no reader is passed generate the reader from the network
        if reader:
            reader_config = reader.generate_config()
        else:    
            description, has_inputs, readers = self.root_node.to_config()        
            reader_config = '\n'.join(r.generate_config() for r in readers)         
        
        tmpl_dict = {
            'DevideId': self.device_id,
            'ModelPath': model_filename,
            'Reader': reader_config,
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

        # if no reader is passed generate the reader from the network
        if reader:
            reader_config = reader.generate_config()
        else:    
            description, has_inputs, readers = self.root_node.to_config()        
            reader_config = '\n'.join(r.generate_config() for r in readers)            

        tmpl_dict = {
            'DevideId': self.device_id,
            'ModelPath': model_filename,
            'PredictOutputFile': output_filename_base,
            'Reader': reader_config,
        }
        return tmpl % tmpl_dict
    
    def _generate_eval_config(self, root_node, reader):
        '''
        Generates the configuration file for write action.
        :param root_node: the node to evaluate. 
        :param reader: the reader used to load the data, None if the network does not have input
        '''        
        model_description, has_input, readers = root_node.to_config()
        if reader:
            readers.append(reader)

        if not has_input and not readers:
            # add dummy input to keep CNTK happy
            # TODO relieve this requirement on CNTK side
            data = [[1, 2], [3, 4]]
            fn = os.path.join(self.directory, 'dummy_input.txt')
            from .reader import NumPyReader
            reader = NumPyReader(data, fn)
            from .ops.cntk1 import Input
            dummy_input_node = Input(2, var_name='dummy_node')
            reader.add_input(dummy_input_node, 0, 2)                        
            model_description += "dummy_node = Input(2, tag='output')"
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
    def train(self, optimizer, reader=None, override_existing = True):
        '''
        Abstract method for the action train.
        :param reader: the reader to use for this action. Alternatively, you
        can attach a reader directly to the input node.
        '''
        pass

    @abstractmethod
    def test(self, reader=None):
        '''
        Abstract method for the action test.
        :param reader: the reader to use for this action. Alternatively, you
        can attach a reader directly to the input node.
        '''
        pass

    @abstractmethod
    def predict(self, reader=None):
        '''
        Abstract method for the action write. It evaluated the trained model on 
        the data provided by the reader.
        :param reader: the reader to use for this action. Alternatively, you
        can attach a reader directly to the input node.

        Returns the predicted output
        '''
        pass

    @abstractmethod
    def eval(self, node, reader=None):
        '''
        Abstract method for the action write. It evaluated the passed node on the
        data provided by the reader.
        :param node: the node to evaluate.
        :param reader: the reader to use for this action. Alternatively, you
        can attach a reader directly to the input node.

        Returns the output generated by `node`
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

        Returns the output generated by the CNTK executable, which is used to
        retrieve the node shapes.
        '''
        filename = os.path.join(self.directory, config_file_name)
        with open(os.path.join(self.directory, filename), "w") as out:
            out.write(config_content)

        try:
            output = subprocess.check_output(
                [CNTK_EXECUTABLE_PATH, "configFile=%s" % filename],
                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(e.output.decode("utf-8"), file=open('error.txt', 'w'))
            raise

        if not output:
            raise ValueError('no output returned')

        return output.decode("utf-8")

    def train(self, optimizer, reader=None, override_existing = True):
        '''
        Run the train action locally.
        :param optimizer: the SGD optimizer to use for training
        :param reader: the reader to use for this action. Alternatively, you
        can attach a reader directly to the input node.
        :param override_existing: if the folder exists already override it
        '''
        config_content = self._generate_train_config(optimizer, reader, override_existing)
        return self._call_cntk(CNTK_TRAIN_CONFIG_FILENAME, config_content)

    def test(self, reader=None):
        '''
        Run the test action locally.
        :param reader: the reader to use for this action. Alternatively, you
        can attach a reader directly to the input node.
        '''
        config_content = self._generate_test_config(reader)
        return self._call_cntk(CNTK_TEST_CONFIG_FILENAME, config_content)

    def predict(self, reader=None):
        '''
        Run the write action locally, use the trained model of this context.
        :param reader: the reader to use for this action. Alternatively, you
        can attach a reader directly to the input node.

        Returns the predicted output
        '''
        config_content = self._generate_predict_config(reader)
        return self._call_cntk(CNTK_PREDICT_CONFIG_FILENAME, config_content)

    '''
    Regular expression to parse the shape information of the nodes out of
    CNTK's output
    '''
    VAR_SHAPE_REGEX = re.compile('^Validating --> (?P<var_name>[^ ]+) = [^>]*> \[(?P<shape>[^]]+)')
    SHAPE_STRIDE_REGEX = re.compile('\{.*?\}')

    @staticmethod
    def _parse_shapes_from_output(output):
        '''
        Parse CNTK's output and look for shape information that is then passed
        as a dictionary {var_name -> shape tuple}
        '''
        var_shape = {}
        for line in output.split('\n'):
            mo = Context.VAR_SHAPE_REGEX.match(line)
            if not mo:
                continue
            var_name, shape = mo.group('var_name'), mo.group('shape')

            shape_list = []
            for x in Context.SHAPE_STRIDE_REGEX.sub('', shape).split('x'):
                x = x.strip()
                if x != '*':
                    shape_list.append(int(x))

            var_shape[var_name] = tuple(shape_list)

        return var_shape
            
    def _eval(self, node, reader):
        # FIXME manually setting the tag to output might have side-effects
        node.tag = 'output'
        config_content = self._generate_eval_config(node, reader)
        output = self._call_cntk(CNTK_EVAL_CONFIG_FILENAME, config_content)
        shapes = Context._parse_shapes_from_output(output)

        out_name = os.path.join(self.directory, CNTK_OUTPUT_FILENAME + '.' + node.var_name)
        data = np.loadtxt(out_name)

        return data, shapes

    def eval(self, node, reader=None):
        '''
        Run the write action locally to evaluate the passed node and returning
        the data it produced.

        :param node: the node to evaluate.
        :param reader: the reader to use for this action. Alternatively, you
        can attach a reader directly to the input node.

        Returns the output generated by `node`
        '''
        if not isinstance(node, ComputationNode):
            raise ValueError('node is not of type ComputationNode, but %s'%type(node))

        data, shapes = self._eval(node, reader)

        expected_size = np.multiply.reduce(shapes[node.var_name])
        expected_shape = shapes[node.var_name]

        receieved_all = data.size == expected_size
        if not receieved_all: 
            # For some reason the CNTK write action has issues with multi-row
            # output. So we have to CNTK reshape it to one row and do it again,
            # but then NumPy reshape using node's expected shape.

            reshaped = NewReshape(node, expected_size)
            data, _ = self._eval(reshaped, reader)

        if not (len(expected_shape)==2 and expected_shape[1] == 1):
            # CNTK outputs e.g. [2 x 1] although it is just a vector.
            # TODO find better way to distinguis between 
            data = data.reshape(expected_shape)

        return data


class ClusterContext(AbstractContext):
    '''
    This is a sub-class of AbstractContext, use it to submit your workloads to the cluster.
    '''
    pass
