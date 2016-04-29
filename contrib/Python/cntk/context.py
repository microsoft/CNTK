# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# TODO: Settle on a centralized location for all the documentation that is in docstrings
# TODO: Take out the saved model from the context

from abc import ABCMeta, abstractmethod
import os
import re
import subprocess
import numpy as np
import shutil as sh

from cntk.graph import ComputationNode
from cntk.utils import get_cntk_cmd, MODEL_INDENTATION
from .utils import cntk_to_numpy_shape
from .utils import with_metaclass
from .reader import InputMap

CNTK_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
CNTK_TRAIN_TEMPLATE_PATH = os.path.join(
    CNTK_TEMPLATE_DIR, "cntk_train_template.cntk")
CNTK_TEST_TEMPLATE_PATH = os.path.join(
    CNTK_TEMPLATE_DIR, "cntk_test_template.cntk")
CNTK_WRITE_TEMPLATE_PATH = os.path.join(
    CNTK_TEMPLATE_DIR, "cntk_write_template.cntk")
CNTK_EVAL_TEMPLATE_PATH = os.path.join(
    CNTK_TEMPLATE_DIR, "cntk_eval_template.cntk")
CNTK_TRAIN_CONFIG_FILENAME = "train.cntk"
CNTK_TEST_CONFIG_FILENAME = "test.cntk"
CNTK_WRITE_CONFIG_FILENAME = "write.cntk"
CNTK_EVAL_CONFIG_FILENAME = "eval.cntk"
CNTK_OUTPUT_FILENAME = "out"

# TODO: add validate method
# TODO: overload action methods to support numpy matrices as inputs
# TODO: overload action methods to have versions that do not need reader
# TODO: clean_up should become a property of train()
# or numpy inputs

_CONTEXT = {}


def get_context(handle='default'):
    # TODO: we need more sanity in the model handling here
    if handle not in _CONTEXT:
        _CONTEXT[handle] = LocalExecutionContext(handle)

    return _CONTEXT[handle]


def get_new_context():
    while True:
        new_handle = str(np.random.random())[2:]
        if new_handle not in _CONTEXT:
            return get_context(new_handle)


class AbstractContext(with_metaclass(ABCMeta, object)):

    '''
    This is the abstract CNTK context. It provides an API to run CNTK actions.
    '''

    def __init__(self, name,
                 device_id=-1,
                 precision="float"):
        
        '''        
        This is the constructor of AbstractContext       
        
        Args:
            name: context name
            device_id: whether to use CPU (-1) or GPU if `device_id>=0', in which case it denotes the GPU index
            precision: either float or double
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
        self.precision = precision
        self.input_nodes = set()

    def __enter__(self):
        _CONTEXT[self.name] = self

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        del _CONTEXT[self.name]

    def _save_file(self, config_file_name, config_content):
        '''
        Writes the content of a config file on disk.

        Args:
            config_file_name (str): the name of the configuration file
            config_content (str): a string containing the configuration
            
        Returns:
            the full path of the saved file
        '''

        filename = os.path.join(self.directory, config_file_name)
        filename = os.path.relpath(filename)

        with open(filename, 'w') as out:
            out.write(config_content)

        return filename
        
    @abstractmethod
    def train(self, root_nodes, optimizer, input_map=None, override_existing=True):
        '''
        Abstract method to run the train action locally.

        Args:
            root_nodes (list): the list of root nodes of the model
            optimizer (instance of `cntk.optimizer.SGDParams'): the SGD optimizer to use for training
            node (:class:`cntk.graph.ComputationNode`): the node to evaluate
            input_map (dict): map from input nodes to `InputMap`
            override_existing (bool): if the folder exists already override it

        Returns:
            output of the CNTK executable training run
        '''
        pass

    @abstractmethod
    def test(self, input_map=None):
        '''
        Abstract method for the action test.

        Args:
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader

        Returns:
            dictionary containing 'SamplesSeen', 'Perplexity', and values for
            objective and evaluation error indexed by their node names
        '''
        pass

    @abstractmethod
    def write(self, input_map=None):
        '''
        Abstract method for the action write. It evaluates the trained model on 
        the data provided by the reader.

        Args:
            node (:class:`cntk.graph.ComputationNode`): the node to evaluate.
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader

        Returns: 
            output generated by `node`
        '''
        pass

    @abstractmethod
    def eval(self, node, input_map=None, backward_pass=False, input_name=None):
        '''
        Abstract method for the action write.  It evaluates `node` on the data
        provided by the reader. This is useful mainly to explore the operators
        and for convenient unit testing.
        
        Args:
            node (:class:`cntk.graph.ComputationNode`): the node to evaluate.
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader
            backward_pass (bool): set to True if you want to output the gradient of a node (backward pass)
            input_name (:class:`cntk.graph.ComputationNode`): if backward_pass is True then input_node should contain the input name that
            the gradient is performed with respect to.

        Returns: 
            output generated by `node`
        '''
        pass

    def _generate_config(self, root_nodes=None, input_map=None):
        '''
        Helper function to create a configuration incorporating all root nodes
        '''

        desc = []
        inputs = set()
        unrolled_nodes = {}
        node_counter = 0

        if not isinstance(root_nodes, list):
            root_nodes = [root_nodes]

        for root_node in root_nodes:
            var_name, node_counter, _desc, _inputs = \
                root_node._to_config(input_map,
                                     desc,
                                     unrolled_nodes,
                                     inputs,
                                     node_counter)

            inputs |= _inputs

        description = "\n".join(desc)

        return description, inputs

    def _generate_train_config(self, root_nodes, optimizer, input_map, override_existing):
        '''
        Generates the configuration file for the train action.

        Args:
            root_nodes (list): the list of root nodes of the model
            optimizer (instance of `cntk.optimizer.SGDParams'): the SGD optimizer to use for training
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader
            override_existing (bool): if the folder exists already override it

        Returns: 
            configuration string
        '''

        if input_map is None:
            input_map = InputMap()

        description, inputs = self._generate_config(root_nodes, input_map)

        tmpl = open(CNTK_TRAIN_TEMPLATE_PATH, "r").read()        

        tmpl_dict = {
            'DevideId': self.device_id,
            'Precision': self.precision,
            'ModelDescription': description,
            'ModelPath': self.model_path,
            'Reader': input_map.generate_config(),
            'SGD': optimizer.generate_config(),
        }

        return tmpl % tmpl_dict

    def _generate_test_config(self, input_map):
        '''
        Generates the configuration file for the test action.

        Args:
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader

        Returns:
            configuration string
        '''
        if input_map is None:
            input_map = InputMap()

        tmpl = open(CNTK_TEST_TEMPLATE_PATH, "r").read()        

        tmpl_dict = {
            'DevideId': self.device_id,
            'Precision': self.precision,
            'ModelPath': self.model_path,
            'Reader': input_map.generate_config(),
        }
        return tmpl % tmpl_dict

    def _generate_write_config(self, input_map):
        '''
        Generates the configuration file for the write action.
        It uses the context's trained model.

        Args:
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader

        Returns:
            configuration string
        '''
        if input_map is None:
            input_map = InputMap()

        tmpl = open(CNTK_WRITE_TEMPLATE_PATH, "r").read()

        tmpl_dict = {
            'DevideId': self.device_id,
            'Precision': self.precision,
            'ModelPath': self.model_path,
            'OutputFile': self.output_filename_base,
            'Reader': input_map.generate_config(),
        }
        return tmpl % tmpl_dict

    def _generate_eval_config(self, root_nodes, input_map=None, node_unit_test=False):
        '''
        Generates the configuration file for write action.

        Args:
            root_nodes (list): the list of root nodes of the model
            node (:class:`cntk.graph.ComputationNode`): the node to evaluate
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader
            node_unit_test (bool): set to True if you want to output the gradient of a node (backward pass)

        Returns: 
            configuration string
        '''
        if input_map is None:
            input_map = InputMap()

        description, inputs = self._generate_config(root_nodes, input_map)

        if len(inputs) == 0:
            # add dummy input to keep CNTK happy
            # TODO relieve this requirement on CNTK side
            #import ipdb;ipdb.set_trace()
            from cntk.ops import input_reader
            dummy_input = input_reader([[[1]]])
            dummy_input.var_name='_dummy_input'
            input_map._add_unmapped(dummy_input)
            desc, _inputs = dummy_input.to_config(input_map)
            description += '\n\n' + desc


        tmpl = open(CNTK_EVAL_TEMPLATE_PATH, "r").read()
        
        tmpl_dict = {
            'DevideId': self.device_id,
            'Precision': self.precision,
            'NodeUnitTest': node_unit_test,
            'OutputFile': self.output_filename_base,
            'ModelDescription': description,
            'Reader': input_map.generate_config(),
        }
        return tmpl % tmpl_dict

class LocalExecutionContext(AbstractContext):

    '''
    This is a sub-class of AbstractContext, use it to run CNTK locally.
    '''

    def __init__(self, name,
                 device_id=-1,
                 precision="float",
                 clean_up=True):
        
        '''        
        This is the constructor of LocalExecutionContext       
        
        Args:
            name: context name
            device_id: whether to use CPU (-1) or GPU if `device_id>=0', in which case it denotes the GPU index
            precision: either float or double
            clean_up: whether the temporary directory should be removed when the context is left        
        '''        
        
        super(self.__class__,self).__init__(name, device_id, precision)
        self.clean_up = clean_up
        self.model_dir = os.path.join(self.directory, 'Models')
        self.model_path = os.path.join(self.model_dir, self.name)
        self.output_filename_base = os.path.join(self.directory, CNTK_OUTPUT_FILENAME)

    def __exit__(self, exc_type, exc_value, exc_tb):
        super(self.__class__, self).__exit__( exc_type, exc_value, exc_tb)
        if self.clean_up:
            sh.rmtree(self.directory)

        
    def _call_cntk(self, config_file_name, config_content):
        '''
        Calls the CNTK executable on the `config_content`.

        Args:
            config_file_name (str): the name of the configuration file
            config_content (str): a string containing the configuration

        Returns:
            the output generated by the CNTK executable, which is used to retrieve the node shapes.
        '''
        
        filename = self._save_file(config_file_name, config_content)

        try:
            output_bytes = subprocess.check_output(
                [get_cntk_cmd(), 'configFile=%s' % filename],
                stderr=subprocess.STDOUT)
            output = output_bytes.decode('utf-8')
            with open(os.path.join(self.directory, 'cntk.log'), 'w') as log:
                log.write(output)

        except subprocess.CalledProcessError as e:
            with open('error.txt', 'w') as f:
                f.write(e.output.decode('utf-8'))
            print("=" * 50)
            print(e.output.decode('utf-8'))
            print("=" * 50)
            raise

        if not output:
            raise ValueError('no output returned')

        return output

    '''
    Regular expression to parse the shape information of the nodes out of
    CNTK's output
    '''
    _VAR_SHAPE_REGEX = re.compile(
        '^Validating --> (?P<var_name>[^ ]+) = [^>]*> \[(?P<shape>[^]]+)')
    _SHAPE_STRIDE_REGEX = re.compile('\{.*?\}')

    @staticmethod
    def _parse_shapes_from_output(output):
        '''
        Parse CNTK's output and look for shape information that is then passed
        as a dictionary {var_name -> shape tuple}

        Args:
            output (str): output from CNTK

        Returns:
            dictionary mapping node names to shapes
        '''
        var_shape = {}
        for line in output.split('\n'):
            mo = LocalExecutionContext._VAR_SHAPE_REGEX.match(line)
            if not mo:
                continue
            var_name, shape = mo.group('var_name'), mo.group('shape')
            # In Debug mode, an additional stride information is printed
            shape = LocalExecutionContext._SHAPE_STRIDE_REGEX.sub('', shape)

            shape_list = []
            for x in shape.split('x'):
                x = x.strip()
                if x == '*':
                    shape_list.append(np.NaN)
                else:
                    shape_list.append(int(x))

            var_shape[var_name] = tuple(shape_list)

        return var_shape

    @staticmethod
    def _sanitized_asarray(data):
        '''
        Data returned from CNTK might contain infinity or NaNs in the form of
        `1.#IND -1.#IND 1.#INF -1.#INF` on Windows or `nan -nan inf -inf` on
        Linux. While the Linux versions are automatically handled by NumPy, the
        Windows versions are not. This function maps those values to NumPy's 
        `nan` and `inf` and returns a NumPy array with dtype=float.

        Args:
            data : Python list of strings 
              Numbers to be converted or inf/nans

        Returns:
            out (ndarray): NumPy array with NaNs and Infs mapped to NumPy versions of them.

        See also:
            http://www.johndcook.com/blog/IEEE_exceptions_in_cpp/
        '''
        try:
            return np.asarray(data, dtype=float)
        except ValueError:

            for i in range(len(data)):
                try:
                    data[i] = float(data[i])
                except ValueError:
                    if data[i].startswith('1.#IND'):
                        data[i] = np.nan
                    elif data[i].startswith('-1.#IND'):
                        data[i] = -np.nan
                    elif data[i].startswith('1.#INF'):
                        data[i] = np.inf
                    elif data[i].startswith('-1.#INF'):
                        data[i] = -np.inf

            return np.asarray(data, dtype=float)

    @staticmethod
    def _parse_result_output(output):
        '''
        Assuming the data has been output using the output format in the
        configuration

            format = [
                # %x = shape, %d = sequenceId
                sequencePrologue=%d\t|w.shape %x\n%d\t|w\s
                sampleSeparator=\n%d\t|w\s
                elementSeparator=\s
            ]

        this method will parse the output of the form

            0	|w.shape 1 1
            0	|w 60.000000
            1	|w.shape 1 2
            1	|w 22.000000
            1	|w 24.000000

        and return a list of tensors.
        '''

        last_seq_idx = None
        list_of_tensors = []
        tensor_seq = []
        shape = None
        for line in output.splitlines():
            parts = line.strip().split('|')

            seq_idx = parts[0].strip()
            payload = parts[1]
            payload_parts = payload.split(' ')
            info = payload_parts[0]
            data = payload_parts[1:]

            if seq_idx != last_seq_idx:
                if not info == 'w.shape':
                    raise ValueError(
                        'expected shape information, but got "%s"' % line)

                if tensor_seq:
                    list_of_tensors.append(
                        LocalExecutionContext._sanitized_asarray(tensor_seq))
                    tensor_seq = []

                last_seq_idx = seq_idx

                shape = cntk_to_numpy_shape(data)

                continue
            else:
                data = LocalExecutionContext._sanitized_asarray(
                    data).reshape(shape, order='F')

            tensor_seq.append(data)

        list_of_tensors.append(np.asarray(tensor_seq))

        return list_of_tensors

    _TEST_RESULT_REGEX = re.compile(
        '(?P<name>[^:]+): [^=]+ = (?P<number>[0-9.]+)')

    @staticmethod
    def _parse_test_result(output):
        result = {}

        PREAMPLE = 'Final Results: Minibatch[1-1]: '
        for line in output.splitlines():

            if not line.startswith(PREAMPLE):
                continue

            line = line[len(PREAMPLE):]

            if not line.startswith('SamplesSeen = '):
                raise ValueError('expected SamplesSeen but got "%s"' % line)

            line = line[len('SamplesSeen = '):]
            number_ends = line.index(' ')
            result['SamplesSeen'] = int(line[:number_ends])
            line = line[number_ends:]

            perplexity_idx = line.index('Perplexity = ')
            result['Perplexity'] = float(
                line[perplexity_idx + len('Perplexity = '):])

            line = line[:perplexity_idx]

            mo = LocalExecutionContext._TEST_RESULT_REGEX.match(line)
            while mo:
                result[mo.group('name').strip()] = float(
                    mo.group('number').strip())
                line = line[mo.span()[1]:]
                mo = LocalExecutionContext._TEST_RESULT_REGEX.match(line)

        return result

    def _calc_expected_shape_and_size(self, node, data, shapes):
        '''
        Calculates the expected shape and size from the CNTK output and the
        retrieved data.

        Args:
            node (:class:`cntk.graph.ComputationNode`): the node that was evaluated.
            data (ndarray): the resulting data from `eval()`
            shapes (dict): dictionary of node names to shape tuples as returned by CNTK

        Returns:
            expected size and shape
        '''

        # We need to reshape it based on CNTK's shape output.

        expected_shape = np.asarray(shapes[node.var_name])

        if sum(np.isnan(expected_shape)) > 1:
            raise ValueError("for node '%s' we received shape '%s', but " +
                             "at most one dimension can be left unspecified." %
                             (node.var_name, expected_shape))

        expected_size = np.multiply.reduce(
            expected_shape[~np.isnan(expected_shape)])
        if sum(np.isnan(expected_shape)) == 1:
            if data.size == expected_size:
                # We received all the data we need, so we have sequences of
                # length 1. For convenience, we ignore it.
                expected_shape = expected_shape[~np.isnan(expected_shape)]

            elif data.size > expected_size:
                # We can fill in the missing dimensions
                missing_dimension = data.size / expected_size
                if int(missing_dimension) != missing_dimension:
                    raise ValueError('could not infer the missing dimensions')

                expected_shape[np.isnan(expected_shape)] = missing_dimension
                expected_size = np.multiply.reduce(expected_shape)
                # Now we have expected_size == data.size
            else:
                raise ValueError('unable to retrieve expected size')

        # Move last dimension to the beginning: this is the time dimension
        #expected_shape = np.roll(expected_shape, 1)

        return expected_shape, expected_size

    def train(self, root_nodes, optimizer, input_map=None, override_existing=True):
        '''
        Run the train action locally.

        Args:
            root_nodes (list): the list of root nodes of the model
            optimizer (instance of `cntk.optimizer.SGDParams'): the SGD optimizer to use for training
            node (:class:`cntk.graph.ComputationNode`): the node to evaluate
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader
            override_existing (bool): if the folder exists already override it

        Returns:
            output of the CNTK executable training run
        '''

        if os.path.exists(self.model_dir):
            if override_existing:
                print("Overriding the existing models")
                sh.rmtree(self.model_dir)
            else:
                raise Exception("Directory '%s' already exists, set the " +
                                "flag override_existing to true if you want to "
                                "override it" % self.directory)

        config_content = self._generate_train_config(
            root_nodes, optimizer, input_map, override_existing)

        return self._call_cntk(CNTK_TRAIN_CONFIG_FILENAME, config_content)

    def test(self, input_map=None):
        '''
        Run the test action locally.

        Args:
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader

        Returns:
            dictionary containing 'SamplesSeen', 'Perplexity', and values for
            objective and evaluation error indexed by their node names
        '''

        config_content = self._generate_test_config(input_map)
        output = self._call_cntk(CNTK_TEST_CONFIG_FILENAME, config_content)

        return LocalExecutionContext._parse_test_result(output)

    def write(self, input_map=None):
        '''
        It evaluates the trained model on the data provided by the reader.

        Args:
            node (:class:`cntk.graph.ComputationNode`): the node to evaluate.
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader

        Returns: 
            output generated by `node`
        '''
        config_content = self._generate_write_config(input_map)
        return self._call_cntk(CNTK_WRITE_CONFIG_FILENAME, config_content)

    def eval(self, node, input_map=None, backward_pass=False, input_name=None):
        '''
        It evaluates `node` on the data provided by the reader. This is useful
        mainly to explore the operators and for convenient unit testing. 
        
        Args:
            node (:class:`cntk.graph.ComputationNode`): the node to evaluate
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader
            backward_pass (bool): set to True if you want to output the gradient of a node (backward pass)
            input_name (:class:`cntk.graph.ComputationNode`): if backward_pass is True then input_node should contain the input name that
            the gradient is performed with respect to.

        Returns: 
            output generated by `node`
        '''
        if not isinstance(node, ComputationNode):
            raise ValueError(
                'node is not of type ComputationNode, but %s' % type(node))

        if backward_pass and input_name is None:
            raise ValueError(
                'an input name is required when backward pass is enabled')

        # Taking note of the original tag of this node to restore it later
        orig_node_tag = node.tag if hasattr(node, 'tag') else None
        node.tag = 'output'

        config_content = self._generate_eval_config(
            node, input_map, backward_pass)
        self._call_cntk(CNTK_EVAL_CONFIG_FILENAME, config_content)

        node.tag = orig_node_tag

        n = input_name.var_name if isinstance(
            input_name, ComputationNode) else input_name
        out_name = os.path.join(self.directory, CNTK_OUTPUT_FILENAME + '.')
        if backward_pass:
            out_name += n + '.grad'
        else:
            out_name += node.var_name

        result_content = open(out_name).read()
        data = LocalExecutionContext._parse_result_output(result_content)

        return data

class DeferredExecutionContext(AbstractContext):

    '''
    This is a sub-class of AbstractContext, use it to generate CNTK configuration,
    that would be executed on different enviroment (e.g., on a cluster) rather than 
    the machine that generated them.
    '''
    
    def __init__(self, name,
                 device_id=-1,
                 precision="float",
                 clean_up=True):
        
        '''        
        This is the constructor of DeferredExecutionContext       
        
        Args:
            name: context name
            device_id: whether to use CPU (-1) or GPU if `device_id>=0', in which case it denotes the GPU index
            precision: either float or double            
        '''        
        
        super(self.__class__,self).__init__(name, device_id, precision)        
        self.model_path = os.path.join("$ModelDir$", self.name)
        self.output_filename_base = os.path.join("$DataDir$", CNTK_OUTPUT_FILENAME)

    def train(self, root_nodes, optimizer, input_map=None, override_existing=True):
        '''
        Prepare the training configuration to be run on a different environment 

        Args:
            root_nodes (list): the list of root nodes of the model
            optimizer (instance of `cntk.optimizer.SGDParams'): the SGD optimizer to use for training
            node (:class:`cntk.graph.ComputationNode`): the node to evaluate
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader
            override_existing (bool): if the folder exists already override it

        Returns:
            training configuration
        '''

        config_content = self._generate_train_config(
            root_nodes, optimizer, input_map, override_existing)

        return self._save_file(CNTK_TRAIN_CONFIG_FILENAME, config_content)

    def test(self, input_map=None):
        '''
        Prepare the testing configuration to be run on a different environment 

        Args:
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader

        Returns:
            testing configuration
        '''

        config_content = self._generate_test_config(input_map)
        self._save_file(CNTK_TEST_CONFIG_FILENAME, config_content)

    def write(self, input_map=None):
        '''
        Prepare the write action configuration to be run on a different environment 

        Args:
            node (:class:`cntk.graph.ComputationNode`): the node to evaluate.
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader

        Returns: 
           write configuration
        '''
        config_content = self._generate_write_config(input_map)
        self._save_file(CNTK_WRITE_CONFIG_FILENAME, config_content)

    def eval(self, node, input_map=None, backward_pass=False, input_name=None):
        '''
        Prepare the evaluation configuration to be run on a different environment. 

        Args:
            node (:class:`cntk.graph.ComputationNode`): the node to evaluate
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader
            backward_pass (bool): set to True if you want to output the gradient of a node (backward pass)
            input_name (:class:`cntk.graph.ComputationNode`): if backward_pass is True then input_node should contain the input name that
            the gradient is performed with respect to.

        Returns: 
            eval configuration
        '''
        if not isinstance(node, ComputationNode):
            raise ValueError(
                'node is not of type ComputationNode, but %s' % type(node))

        if backward_pass and input_name is None:
            raise ValueError(
                'an input name is required when backward pass is enabled')

        node.tag = 'output'

        config_content = self._generate_eval_config(
            node, input_map, backward_pass)
        self._save_file(CNTK_EVAL_CONFIG_FILENAME, config_content)
        
