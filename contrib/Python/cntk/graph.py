from abc import ABCMeta, abstractmethod
import numpy as np

# Workaround until we have switched to Anaconde with scipy support
#import scipy.sparse as sparse 
class sparse(object):
    @staticmethod
    def issparse(obj):
        return hasattr(obj, 'todense')

from .utils import MODEL_INDENTATION

def _tuple_to_cntk_shape(shape):
    return ':'.join(str(v) for v in shape)


class ComputationNode(object):

    '''
    Base class for all nodes and operators. Provides a NumPy-like interface
    with operators that are converted to CNTK operators.
    '''

    def __init__(self, name, params=None, var_name=None, reader=None):
        if not isinstance(name, str):
            raise ValueError(
                "Parameter 'name' has to be a string and not '%s'" % type(name))
        if var_name is not None and not isinstance(var_name, str):
            raise ValueError(
                "Parameter 'var_name' has to be a string and not '%s'" % type(var_name))

        self.name = name
        self.params = params
        self.var_name = var_name
        self.consumers = []
        for p in self.params:
            if hasattr(p, 'consumers'):
                p.consumers.append(self)

        self.reader = None

    def _is_input(self):
        return isinstance(self, InputComputationNodeBase)

    def __add__(self, other):
        if not isinstance(other, ComputationNode):
            # TODO: in case of non-scalars we have to pull in a reader
            other = constant(other)
        return Plus(self, other)

    def __radd__(self, other):
        if not isinstance(other, ComputationNode):
            # TODO: in case of non-scalars we have to pull in a reader
            other = constant(other)
        return Plus(other, self)

    def __sub__(self, other):
        if not isinstance(other, ComputationNode):
            # TODO: in case of non-scalars we have to pull in a reader
            other = constant(other)
        return Minus(self, other)

    def __rsub__(self, other):
        if not isinstance(other, ComputationNode):
            # TODO: in case of non-scalars we have to pull in a reader
            other = constant(other)
        return Minus(other, self)

    def __mul__(self, other):
        if not isinstance(other, ComputationNode):
            # TODO: in case of non-scalars we have to pull in a reader
            other = constant(other)
        return ElementTimes(self, other)

    def __rmul__(self, other):
        if not isinstance(other, ComputationNode):
            # TODO: in case of non-scalars we have to pull in a reader
            other = constant(other)
        return ElementTimes(other, self)

    def __matmul__(self, other):
        if not isinstance(other, ComputationNode):
            # TODO: in case of non-scalars we have to pull in a reader
            other = constant(other)
        # NOTE supported in Python 3.5
        return Times(self, other)

    def __rmatmul__(self, other):
        if not isinstance(other, ComputationNode):
            # TODO: in case of non-scalars we have to pull in a reader
            other = constant(other)
        # NOTE supported in Python 3.5
        return Times(other, self)

    def __abs__(self):
        return Abs(self)

    def __getitem__(self, so):
        if so.stop == None:
            raise ValueError('The stop index has to be provided')

        if isinstance(so, int):
            return RowSlice(self, so, 1)

        elif isinstance(so, slice):
            if so.step not in {1, None}:
                raise ValueError("RowSlice does not support strides")

            start = so.start or 0

            return RowSlice(self, start, so.stop - start)

    # TODO more __operators__

    def _get_cntk_param_string(self, param_variable_names=None):
        return ", ".join(param_variable_names)

    def __str__(self):
        return "%s / params=%s" % (self.name, self.params)

    def _param_to_brainscript(self, p_name, p_value):
        if isinstance(p_value, bool):
            p_value = str(p_value).lower()
        elif isinstance(p_value, str):
            p_value = "'%s'" % p_value
        elif type(p_value) in [list, tuple]:
            # FIXME here we assume that all dims are of TensorShape
            if p_name in ['dims', 'inputs']:
                p_value = _tuple_to_cntk_shape(p_value)
            else:
                raise ValueError('Sequence initialization is only allowed for' +
                                 ' parameters dims and not "%s"' % p_name)
        else:
            p_value = str(p_value)

        if p_name in self.params_with_defaults:
            param = '%s=%s' % (p_name, p_value)
        else:
            param = p_value

        return param


    def _is_forward_ref(self, p_name, p_value): 
        '''
        Although the unrolled graph is a DAG, when we specify recurrence we
        naturally have loops. We can resolve this by using forward references.
        This method is checking whether the particular name and value of this
        instance are actually one of those forward references.
        '''
        is_loop_node = self.name in ('Delay', 'PastValue', 'FutureValue')
        return is_loop_node and p_name == 'input' and isinstance(p_value, str)

    def _to_config_recursively(self, desc, unrolled_nodes, inputs,
                               readers, node_counter=0):
        param_variable_names = []
        if self.params:
            for p_name in self.params:
                p_value = self.__dict__[p_name]
                if hasattr(p_value, '_to_config') and p_name or \
                        p_name == 'inputs':
                        # TODO this is under the assumption that RowStack's
                        # inputs parameter gets a tuple of inputs

                    if p_name == 'inputs' and isinstance(self, RowStack):
                        # Special treatment for special operator.
                        # Used like RowStack(v0:v1:v2)
                        inputs_param = p_value
                    else:
                        inputs_param = [p_value]

                    input_nodes_vars = []
                    for pv in inputs_param:
                        if pv in unrolled_nodes:
                            # We have seen this node already, so just retrieve its
                            # name.
                            child_var = unrolled_nodes[pv]
                        else:
                            child_var, node_counter, child_desc = pv._to_config_recursively(
                                desc, unrolled_nodes, inputs, readers, node_counter)
                            unrolled_nodes[pv] = child_var
                        input_nodes_vars.append(child_var)

                    param_variable_names.append(
                        _tuple_to_cntk_shape(input_nodes_vars))
                else:
                    if self._is_forward_ref(p_name, p_value):
                        # We have a forward reference to a node that will be
                        # later on defined. p_value is the var_name of the
                        # later defined node.
                        param_variable_names.append(p_value)
                    else:
                        param_variable_names.append(
                            self._param_to_brainscript(p_name, p_value))

        if self.reader:
            readers.add(self.reader)

        if self._is_input():
            inputs.add(self)

        if hasattr(self, 'tag') and 'tag' not in self.params:
            param_variable_names.append("tag='%s'" % self.tag)

        self.var_name = self.var_name or "v%i" % node_counter
        node_counter += 1

        params = self._get_cntk_param_string(param_variable_names)

        line = ' ' * MODEL_INDENTATION + \
            "%s = %s(%s)" % (self.var_name, self.name, params)
        desc.append(line)

        return self.var_name, node_counter, desc

    def _to_config(self):
        '''
        Helper method to generate the CNTK configuration for this node.
        '''
        unrolled_nodes = {}
        inputs = set()
        readers = set()
        var_name, node_counter, desc = self._to_config_recursively(
            desc=[],
            unrolled_nodes=unrolled_nodes,
            inputs=inputs,
            readers=readers)

        return var_name, node_counter, desc, len(inputs) > 0, readers

    def _dedupe_readers(self, readers):
        import copy
        readers_map = {}
        for r in readers:
            filename = r['FileName']
            if filename in readers_map:
                readers_map[filename].inputs_def.extend(r.inputs_def)
            else:
                readers_map[filename] = copy.deepcopy(r)

        return [r for r in readers_map.values()]

    def to_config(self):
        '''
        Generate CNTK configuration for this node including the configuration
        for all dependent child nodes.
        '''
        var_name, node_counter, desc, has_inputs, readers = self._to_config()

        return "\n".join(desc), has_inputs, self._dedupe_readers(readers)


class InputComputationNodeBase(ComputationNode, metaclass=ABCMeta):

    '''
    Base class for all non-image input nodes nodes and operators. Provides methods to attach
    a reader to an input node
    '''

    def attach_text_format_reader(self, filename, input_alias=None, format='dense'):
        '''
        attach a TextFormatReader to the node
        '''
        self.reader = CNTKTextFormatReader(filename)
        self.reader.add_input(self, input_alias, self.dims, format)

    def attach_uci_fast_reader(self,
                               filename,
                               input_start,
                               islabel=False,
                               num_label_cols=None,
                               label_mapping_file=None,
                               custom_delimiter=None):
        '''
        attach a UCIFastReader to the node
        '''
        self.reader = UCIFastReader(filename, custom_delimiter)

        if islabel:
            self.reader.add_input(
                self, input_start, num_label_cols, self.dims, label_mapping_file)
        else:
            self.reader.add_input(self, input_start, self.dims)


class ImageInputComputationNodeBase(ComputationNode, metaclass=ABCMeta):

    '''
    Base class for all image input nodes nodes and operators. Provides methods to attach
    a reader to an input node
    '''

    def attach_image_reader(self, filename, **kw):
        '''
        attach a TextFormatReader to the node
        '''
        raise NotImplementedError

# importing after defining ComputationNode to work around circular imports
from cntk.ops.cntk1 import *
# to have a separate namespace when we want to override below
from cntk.ops import cntk1 as cntk1_ops
from .reader import UCIFastReader, CNTKTextFormatReader

# redefine some operators to work with NumPy and sequences as input


def _dense_seq_to_str(seq):
    return ' '.join(seq.astype(np.str))


def _sparse_seq_to_str(seq):
    # return ' '.join('%s:%s'%(k,seq[k]) for k in sorted(seq.items()))
    raise NotImplementedError


def _seq_to_text_format(sequences, alias):
    '''
    `sequences` is a NumPy array
    '''
    if not alias or not isinstance(alias, str):
        raise ValueError('alias is missing')

    first_elem = sequences[0]
    if isinstance(first_elem, np.ndarray):
        seq_to_str = _dense_seq_to_str
    elif sparse.issparse(first_elem):
        seq_to_str = _sparse_seq_to_str
    else:
        raise ValueError(
            'sequence elements have to be of type numpy.ndarray (dense) or dictionary (sparse), you gave "%s"' % str(first_elem))

    lines = []
    for idx, seq in enumerate(sequences):
        lines.append('%i|%s %s' % (idx, alias, seq_to_str(seq)))

    return '\n'.join(lines)


def _get_constant_node(value, **kw):
    '''
    This function creates a node that represents `value` as a constant tensor
    in the graph.

    To be as generic as possible, we 
     - flatten the data 
     - initialize a LearnableParameter operator with it
     - ensure that the graph does not backprob to it.  
     - Finally we to reshape it.
    '''

    # FIXME We need to better manage the context. How can we get hold
    # of the overall context without having to always pass it
    # explicitly?

    from cntk.context import get_context
    import tempfile

    # We have to use NamedTemporaryFile and close it, because when using the
    # obvious first choice, mkstemp(), would later fail in cntk.exe because the
    # file would still be locked.
    # TODO make it same filename as alias
    tf = tempfile.NamedTemporaryFile(
        prefix='_param_', suffix='.txt', dir=get_context().directory, delete=False)
    tf.close()

    if isinstance(value, list):
        value = np.asarray(value)

    if len(value.shape) == 1:
        # 1D list: interpret as one scalar per sample
        value = value[:, np.newaxis]

    if sparse.issparse(value):
        raise ValueError('only dense data is supported')

    with open(tf.name, 'w') as f:
        # TODO value.ravel() ?
        np.ndarray.flatten(value).tofile(f, sep='\n')

    size = np.multiply.reduce(value.shape[:])

    # The var_name specified by the user should be set to the operator that
    # is finally returned, which is the shape node.
    var_name = kw.pop('var_name', None)

    from cntk.reader import CNTKTextFormatReader
    param_node = cntk1_ops.LearnableParameter(
        size,
        1,
        learningRateMultiplier=0.0,
        init='fromFile',
        initFromFilePath=tf.name,
        **kw)

    reshape_node = cntk1_ops.NewReshape(param_node,
                                        dims=value.shape,
                                        var_name=var_name)

    return reshape_node


def _get_input_node(value, **kw):
    # FIXME We need to better manage the context. How can we get hold
    # of the overall context without having to always pass it
    # explicitly?

    from cntk.context import get_context
    import tempfile

    # We have to use NamedTemporaryFile and close it, because the obvious first
    # choice, mkstemp(), would later fail in cntk.exe because the file would
    # still be locked.
    tf = tempfile.NamedTemporaryFile(prefix='_input_', suffix='.txt',
                                     dir=get_context().directory, delete=False)
    tf.close()

    if isinstance(value, list):
        value = np.asarray(value)

    if len(value.shape) == 1:
        # 1D list: interpret as one scalar per sample
        value = value[:, np.newaxis]

    if 'alias' in kw:
        alias = kw['alias']
        del kw['alias']  # don't confuse with constructor's parameters
    else:
        # TODO make sure we don't have clashes
        alias = '_I_%i' % np.random.randint(1000)

    with open(tf.name, 'w') as f:
        f.write(_seq_to_text_format(value, alias))

    from cntk.reader import CNTKTextFormatReader
    input_node = cntk1_ops.Input(value.shape, **kw)
    input_node.reader = CNTKTextFormatReader(tf.name)
    # In case we have the shape (2,3), which will be initialized at Input() as
    # '2:3', we have 2*3 = 6 dimensions when flattened out for the reader. Note
    # that the first dimension is the sample.
    dims = np.multiply.reduce(value.shape[:])
    input_node.reader.add_input(input_node, alias, dims)

    return input_node


def is_sequence(data):
    '''
    Checks whether the data is a CNTK sequence, which is expressed in Python as
    a list of varying sized NumPy objects.
    '''
    is_list = isinstance(data, list)
    return is_list and len(data) > 0 and isinstance(data[0], np.ndarray)


def is_tensor(data):
    '''
    Checks whether the data is a tensor.
    '''
    if isinstance(data, np.ndarray):
        return True

    if not isinstance(data, list):
        return False

    while len(data) > 0:
        # All but the innermost dimension's values have to be lists
        try:
            data[0][0]
        except:
            # We reached the innermost dimension
            break

        if not isinstance(data[0], list):
            return False

        data = data[0]

    return True


def input(value, **kw):
    '''
    Defining Input as a factory override that creates either a Constant()
    operator or an Input() operator based on the type of the `value`.

    In case the `value` is a scalar, a normal CNTK Constant() operator is
    returned.

    In case the `value` is a list of NumPy arrays, a CNTK Input() operator is
    returned, interpreting every element as a sequence of tensors.

    In case the `value` is a NumPy array or list of lists, a CNTK Input()
    operator is returned, interpreting it as a dense tensor.

    Non-scalar values are interpreted as sparse when they contain a colon.
    '''
    if is_sequence(value) or is_tensor(value):
        return _get_input_node(value, **kw)
    else:
        raise ValueError('value type is not supported: %s' % type(value))


def constant(value, **kw):
    '''
    Defining Constant as a factory override that creates either a Constant()
    operator or an Input() operator based on the type of the `value`.

    In case the `value` is a scalar, a normal CNTK Constant() operator is
    returned.

    In case the `value` is a list of NumPy arrays, a CNTK Input() operator is
    returned, interpreting every element as a sequence of tensors.

    In case the `value` is a NumPy array or list of lists, a CNTK Input()
    operator is returned, interpreting it as a dense tensor.

    Non-scalar values are interpreted as sparse when they contain a colon.
    '''
    if np.isscalar(value):
        return cntk1_ops.Constant(value, **kw)
    else:
        if is_tensor(value):
            return _get_constant_node(value, **kw)
        else:
            raise ValueError('value type is not supported: %s' % type(value))
