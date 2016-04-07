from abc import ABCMeta, abstractmethod
import os
import re
import sys
import subprocess
import numpy as np
import shutil as sh

from cntk.graph import ComputationNode
from cntk.ops.cntk1 import NewReshape
from cntk.utils import CNTK_EXECUTABLE_PATH, MODEL_INDENTATION
from .utils import cntk_to_numpy_shape, dedupe_readers

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
                 root_nodes=None,
                 clean_up=True,
                 node_unit_test=False):
        '''
        AbstractContext Constructer

        :param name: context name
        :param graph: the computational graph to be used for training, testing and prediction        
        :param device_id: whether to use CPU or a specific GPU. -1 for CPU larger values
        :param root_nodes: list of top nodes of the graph or single node itself
        :param clean_up: whether the temporary directory should be removed when the context is left
        are the GPUs indices.        
        :param node_unit_test: set to True if you want to output the gradient of a node (backward pass)

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
        if root_nodes is None:
            self.root_nodes = None
        else:
            self.root_nodes = root_nodes if isinstance(root_nodes, list) else [root_nodes]
        self.node_unit_test= node_unit_test

    def __enter__(self):
        _CONTEXT[self.name] = self

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        del _CONTEXT[self.name]

        if self.clean_up:
            sh.rmtree(self.directory)

    def _generate_config(self, root_nodes=None):
        '''
        Helper function to create a configuration incorporating all root nodes
        '''
        has_inputs = False

        desc = []
        inputs = set()
        readers = set() 
        unrolled_nodes = {}
        node_counter = 0
        dep_inputs = tuple()
        reconciled_cache = {}

        if root_nodes is None:
            root_nodes = self.root_nodes
        elif not isinstance(root_nodes, list):
            root_nodes = [root_nodes]

        for root_node in root_nodes:
            var_name, node_counter, _desc, _has_inputs, _readers, _dep_inputs = \
                root_node._to_config(desc, 
                        unrolled_nodes, 
                        inputs,
                        readers, 
                        dep_inputs,
                        node_counter, reconciled_cache)

            has_inputs |= _has_inputs
            readers |= _readers
            dep_inputs += _dep_inputs

        description = "\n".join(desc)

        return description, has_inputs, dedupe_readers(readers)

    def _generate_train_config(self, optimizer, reader, override_existing):
        '''
        Generates the configuration file for the train action.
        :param optimizer: the SGD optimizer to use for training
        :param reader: the reader to use for reading the data
        :param override_existing: if the folder exists already override it
        '''

        model_dir = os.path.join(self.directory, 'Models')
        if os.path.exists(model_dir):
            if override_existing:
                print("Overriding the existing models")
                sh.rmtree(model_dir)
            else:
                raise Exception("Directory '%s' already exists, set the " + 
                        "flag override_existing to true if you want to "
                        "override it" % self.directory)

        tmpl = open(CNTK_TRAIN_TEMPLATE_PATH, "r").read()
        model_filename = os.path.join(model_dir, self.name)
        description, has_inputs, readers = self._generate_config()
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
            description, has_inputs, readers = self._generate_config()
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
        output_filename_base = os.path.join(
            self.directory, 'Outputs', self.name)

        # if no reader is passed generate the reader from the network
        if reader:
            reader_config = reader.generate_config()
        else:
            description, has_inputs, readers = self._generate_config()
            reader_config = '\n'.join(r.generate_config() for r in readers)

        tmpl_dict = {
            'DevideId': self.device_id,
            'ModelPath': model_filename,
            'PredictOutputFile': output_filename_base,
            'Reader': reader_config,
        }
        return tmpl % tmpl_dict

    def _generate_eval_config(self, root_nodes, reader):
        '''
        Generates the configuration file for write action.
        :param root_nodes: the node to evaluate. 
        :param reader: the reader used to load the data, None if the network does not have input
        '''
        description, has_inputs, readers = self._generate_config(root_nodes)
        if reader:
            readers.append(reader)

        if not has_inputs and not readers:
            # add dummy input to keep CNTK happy
            # TODO relieve this requirement on CNTK side
            data = [[1, 2], [3, 4]]
            fn = os.path.join(self.directory, 'dummy_input.txt')
            from .reader import NumPyReader
            reader = NumPyReader(data, fn)
            from .ops.cntk1 import Input
            dummy_input_node = Input(2, var_name='dummy_node')
            reader.add_input(dummy_input_node, 0, 2)
            description += "\n" + " "*MODEL_INDENTATION + "dummy_node = Input(2, tag='output')"
            readers.append(reader)

        tmpl = open(CNTK_EVAL_TEMPLATE_PATH, "r").read()
        output_filename = os.path.join(self.directory, CNTK_OUTPUT_FILENAME)
        tmpl_dict = {
            'DevideId': self.device_id,
            'NodeUnitTest': self.node_unit_test,
            'OutputFile': output_filename,
            'ModelDescription': description,
            'Reader': '\n'.join(r.generate_config() for r in readers),
        }
        return tmpl % tmpl_dict

    @abstractmethod
    def train(self, optimizer, reader=None, override_existing=True):
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
        with open(os.path.join(self.directory, filename), 'w') as out:
            out.write(config_content)

        try:
            output_bytes = subprocess.check_output(
                [CNTK_EXECUTABLE_PATH, 'configFile=%s' % filename],
                stderr=subprocess.STDOUT)
            output = output_bytes.decode('utf-8')
            with open(os.path.join(self.directory, 'cntk.log'), 'w') as log:
                log.write(output)

        except subprocess.CalledProcessError as e:
            print(e.output.decode('utf-8'), file=open('error.txt', 'w'))
            raise

        if not output:
            raise ValueError('no output returned')

        return output

    def train(self, optimizer, reader=None, override_existing=True):
        '''
        Run the train action locally.
        :param optimizer: the SGD optimizer to use for training
        :param reader: the reader to use for this action. Alternatively, you
        can attach a reader directly to the input node.
        :param override_existing: if the folder exists already override it
        '''

        config_content = self._generate_train_config(
            optimizer, reader, override_existing)
        return self._call_cntk(CNTK_TRAIN_CONFIG_FILENAME, config_content)

    def test(self, reader=None):
        '''
        Run the test action locally.
        :param reader: the reader to use for this action. Alternatively, you
        can attach a reader directly to the input node.
        '''
        config_content = self._generate_test_config(reader)
        output = self._call_cntk(CNTK_TEST_CONFIG_FILENAME, config_content)

        return Context._parse_test_result(output)


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
    VAR_SHAPE_REGEX = re.compile(
        '^Validating --> (?P<var_name>[^ ]+) = [^>]*> \[(?P<shape>[^]]+)')
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
            # In Debug mode, an additional stride information is printed
            shape = Context.SHAPE_STRIDE_REGEX.sub('', shape)

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
            parts = line.split('|')

            seq_idx = parts[0].strip()
            payload = parts[1]
            info, *data = payload.split(' ')

            if seq_idx != last_seq_idx:
                if not info == 'w.shape':
                    raise ValueError('expected shape information, but got "%s"'%line) 

                if tensor_seq:
                    list_of_tensors.append(np.asarray(tensor_seq))
                    tensor_seq = []

                last_seq_idx = seq_idx

                shape = cntk_to_numpy_shape(data)

                continue
            else:
                data = np.asarray(data, dtype=float).reshape(shape)

            tensor_seq.append(data)

        list_of_tensors.append(np.asarray(tensor_seq))

        return list_of_tensors

    TEST_RESULT_REGEX = re.compile('(?P<name>[^:]+): [^=]+ = (?P<number>[0-9.]+)')

    @staticmethod
    def _parse_test_result(output):
        result = {}

        PREAMPLE = 'Final Results: Minibatch[1-1]: '
        for line in output.splitlines():

            if not line.startswith(PREAMPLE):
                continue

            line = line[len(PREAMPLE):]

            if not line.startswith('SamplesSeen = '):
                raise ValueError('expected SamplesSeen but got "%s"'%line)

            line = line[len('SamplesSeen = '):]
            number_ends = line.index(' ')
            result['SamplesSeen'] = int(line[:number_ends])
            line = line[number_ends:]

            perplexity_idx = line.index('Perplexity = ')
            result['Perplexity'] = float(line[perplexity_idx+len('Perplexity = '):])

            line = line[:perplexity_idx]

            mo = Context.TEST_RESULT_REGEX.match(line)
            while mo:
                result[mo.group('name').strip()] = float(mo.group('number').strip())
                line = line[mo.span()[1]:]
                mo = Context.TEST_RESULT_REGEX.match(line)

        return result

    def _calc_expected_shape_and_size(self, node, data, shapes):
        '''
        Calculates the expected shape and size from the CNTK output and the
        retrieved data.

        :param node: the node that was evaluated.
        :param data: the resulting data from `eval()`
        :param shapes: dictionary of node names to shape tuples

        Returns the expected size and shape
        '''

        # We got a single-dimensional array back, so we have to check whether
        # we need to reshape it based on CNTK's shape output.

        expected_shape = np.asarray(shapes[node.var_name])

        if sum(np.isnan(expected_shape))>1:
            raise ValueError("for node '%s' we received shape '%s', but " +
                    "at most one dimension can be left unspecified."%\
                            (node.var_name, expected_shape))

        expected_size = np.multiply.reduce(expected_shape[~np.isnan(expected_shape)])
        if sum(np.isnan(expected_shape))==1:
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
            raise ValueError(
                'node is not of type ComputationNode, but %s' % type(node))

        # Taking note of the original tag of this node to restore it later
        orig_node_tag = node.tag if hasattr(node, 'tag') else None
        node.tag = 'output'

        config_content = self._generate_eval_config(node, reader)
        output = self._call_cntk(CNTK_EVAL_CONFIG_FILENAME, config_content)

        node.tag = orig_node_tag

        shapes = Context._parse_shapes_from_output(output)

        out_name = os.path.join(
            self.directory, CNTK_OUTPUT_FILENAME + '.' + node.var_name)
        #data = np.loadtxt(out_name)
        result_content = open(out_name).read()
        data = Context._parse_result_output(result_content)

        return data


class ClusterContext(AbstractContext):

    '''
    This is a sub-class of AbstractContext, use it to submit your workloads to the cluster.
    '''
    pass
