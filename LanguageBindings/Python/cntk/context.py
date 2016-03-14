from abc import ABCMeta, abstractmethod
import os
import subprocess
import numpy as np


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

    """This is the abstract CNTK context. It provides an API to run CNTK actions
    """

    def __init__(self, name,
                 graph=None,
                 optimizer=None,
                 device_id=-1,
                 root_node=None,
                 clean_up=True):
        """AbstractContext Constructer

        :param name: context name
        :param graph: the computational graph to be used for training, testing and prediction
        :param optimizer: the SGD optimizer to use for training
        :param device_id: whether to use CPU or a specific GPU. -1 for CPU larger values
        :param root_node: the top node of the graph
        :param clean_up: whether the temporary directory should be removed when the context is left
        are the GPUs indices.

        """
        if isinstance(name, str):
            tmpdir = name
        else:
            tmpdir = id(name)

        self.directory = os.path.abspath('_cntk_%s' % tmpdir)

        if os.path.exists(self.directory):
            print("Directory '%s' already exists - overwriting data." %
                  self.directory)
        else:
            os.mkdir(self.directory)

        self.name = name
        self.macros = []
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

    def add_input(self, node):
        self.input_nodes.add(node)

    def to_description(self, node, **kw):
        return node.to_description()

    def root_to_description(self, **kw):
        return self.root_node.to_description()

    def add_macro(self, path):
        """Add a macro file to be referenced from all configurations of this context.
        :param path: path of the macro file.    
        """
        self.macros.append(path)

    def _generate_train_config(self, input_map):
        """Generates the configuration file for the train action.
        """
        tmpl = open(CNTK_TRAIN_TEMPLATE_PATH, "r").read()
        reader_config = reader.generate_config()
        model_filename = os.path.join(self.directory, 'Models', self.name)
        tmpl_dict = {
            'DevideId': self.device_id,
            'ModelDescription': self.root_to_description(),
            'ModelPath': model_filename,
            'Reader': self._generate_reader_config(input_map),
            'SGD': self.optimizer.generate_config(),
        }
        return tmpl % tmpl_dict

    def _generate_test_config(self, reader):
        """Generates the configuration file for the test action.
        """
        raise NotImplementedError

    def _generate_predict_config(self, reader):
        """Generates the configuration file for the write action.
        It uses the context's trained model.
        """
        raise NotImplementedError

    def _check_input_is_assigned(self, input_map):
        not_assigned = []

        for node in self.input_nodes:
            if node not in input_map:
                not_assigned.append('%s/%s' % (node.var_name, node.name))

        if not_assigned:
            raise ValueError('Cannot create the configuration, because the ' +
                             'following input nodes are missing corresponding input readers: ' +
                             ", ".join(not_assigned))

    def _generate_reader_config(self, input_map):
        '''
        Generates the reader configuration including the dimensions of the
        individual nodes that will be fed by the readers.
        '''
        # Putting readers into a set to make sure every reader is configured
        # only once.
        readers = set()
        node_dimensions = []

        # TODO: all we need to configure a reader block for an input is dims and name
        # we do not need yet a reader on to. it is getting complex between many
        # readers, input_nodes and input_map
        for node, (reader, dims) in input_map.items():
            if not node.var_name:
                raise ValueError(
                    "Node '%s' does not have a variable name assigned yet." % str(node))

            start_index, num_dims = dims
            node_dimensions.append('''
            %s = [
               start=%i
               dim=%i
           ]''' % (node.var_name, start_index, num_dims))

            readers.add(reader)

        node_dimensions = '\n'.join(node_dimensions)
        reader_configs = '\n'.join([r.generate_config() for r in readers])

        return "%s\n\n%s" % (node_dimensions, reader_configs)

    # TODO: re-implement with a propoer design in mind.
    def _generate_eval_config(self, root_node, input_map):
        """Generates the configuration file for write action.
        :param root_node: the node to evaluate. 
        :param input_map: mapping of input node to (reader, (start_dim, num_dim))
        """
        self._check_input_is_assigned(input_map)

        # TODO factor out reader config output so that train/test can use it
        model_description = root_node.to_description()

        if not self.input_nodes:
            # add dummy input to keep CNTK happy
            # TODO relieve this requirement

            data = [[1, 2], [3, 4]]
            fn = os.path.join(self.directory, 'dummy_input.txt')
            from .reader import NumPyReader
            reader = NumPyReader(data, fn)
            from .ops import Input
            dummy_input_node = Input(2, ctx=self)
            dummy_input_node.var_name = 'dummy_node'
            input_map = {dummy_input_node: (reader, (0, 2))}
            model_description += "\ndummy_node=Input(2, tag='output')"

        tmpl = open(CNTK_EVAL_TEMPLATE_PATH, "r").read()
        output_filename = os.path.join(self.directory, CNTK_OUTPUT_FILENAME)
        tmpl_dict = {
            'DevideId': self.device_id,
            'Reader': self._generate_reader_config(input_map),
            'OutputFile': output_filename,
            'ModelDescription': model_description
        }
        return tmpl % tmpl_dict

    @abstractmethod
    def train(self, input_map):
        """Abstract method for the action train.
        :param input_map: mapping of input node to (reader, (start_dim, num_dim))
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

        subprocess.check_call(
            [CNTK_EXECUTABLE_PATH, "configFile=%s" % filename])

    def train(self, input_map):
        """Run the train action locally.
        :param input_map: mapping of input node to (reader, (start_dim, num_dim))
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
        # FIXME manually setting the tag to output might have side-effects
        node.tag = 'output'
        config_content = self._generate_eval_config(node, input_map)
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

    """This is a sub-class of AbstractContext, use it to submit your workloads to the cluster.
    """
    pass
