# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from abc import ABCMeta, abstractmethod
import collections
import numpy as np

from .utils import with_metaclass, MODEL_INDENTATION, tensors_to_text_format
from .graph import ComputationNode


class AbstractReader(with_metaclass(ABCMeta)):

    """Abstract class that represents a reader.
    """

    # required so that instances can be put into a set
    def __hash__(self): return hash(id(self))

    def __eq__(self, x): return x is self

    def __ne__(self, x): return x is not self


class UCIFastReader(AbstractReader):

    """`Deprecated` - A UCIFastReader for one input node. Please switch to
    :class:`cntk.reader.CNTKTextFormatReader`.

    Note that the dimensions are not inferred from the input node's shape,
    because in case of a label node the dimension does not match the shape
    which would be (``numOfClasses``, 1).

    Args:
        filename (str): the name of the file where the data is stored
        custom_delimiter (str): what delimiter is used to separate columns, specify it in case it neither tab nor white spaces.
        input_start (int): the start column   
        input_dim (int): the number of columns
        num_of_classes (int): the number of classes
        label_mapping_file (str): the mapping file path, it can be simply with all the possible classes, one per line
    """

    def __init__(self, filename, input_start, input_dim,
                 num_of_classes=None, label_mapping_file=None,
                 custom_delimiter=None):
        ''' Reader constructor. 
        '''

        self.filename = filename
        self.custom_delimiter = custom_delimiter
        self.input_start = input_start
        self.input_dim = input_dim
        self.num_of_classes = num_of_classes
        self.label_mapping_file = label_mapping_file

    def _to_aggregate_form(self, input_node):
        r = UCIFastReaderAggregator(self.filename, self.custom_delimiter)
        r.add_input(input_node, self.input_start, self.input_dim,
                    self.num_of_classes, self.label_mapping_file)
        return r


class CNTKTextFormatReader(AbstractReader):

    """A CNTKTextFormatReader for one input node that supports sequences. For a
    full format definition please see
    https://github.com/Microsoft/CNTK/wiki/CNTKTextFormat-Reader.

    Example:
       The following example encodes two samples, one has a sequence of one
       scalar, while the second has a sequence of two scalars::

           0\t|I 60.0
           1\t|I 22.0
           1\t|I 24.0

       The ``I`` is the alias, which would be used to connect the data to the
       input node. Let's say the above data is stored in ``data.txt``, you would
       set up the reader as follows::

           r = CNTKTextFormatReader('data.txt')

       and then later use ``r`` to map the alias ``I`` to the input node. Let's say
       the input node is called ``in_node`` in your code, you would say::
           
           ctx.train(..., input_map=r.map(in_node, alias='I', dim=1, format='dense'))

       The alias is required, because using this format you can set up
       multiple inputs per sample::

           0\t|I 60.0 |W 1 2
           1\t|I 22.0 |W 0 0
           1\t|I 24.0

       In this example, the first sample has ``I`` and ``W`` defined, while
       the second sample has ``I`` for both sequence elements, while ``W`` is
       providing only one data point for the full sequence. This is useful, if
       e.g. a sentence being a sequence of varying number of words is tagged
       with a label.

       If your data is in matrix format (one column per feature), you can use
       `uci2ctf.py <https://github.com/Microsoft/CNTK/blob/master/Scripts/uci2ctf.py>`_
       to convert it to the CNTKTextFormatReader format.

    Args:
        filename (str): path to the input file to read from
        randomize (bool): whether the input should be randomized. ``True`` (default): randomly shuffle the input data; ``False``: do not shuffle at all and take the input in the order it was read
        skip_sequence_ids (bool): if ``True``, the sequence ID will be ignored and every line will be treated as a separate sequence
        max_errors (int): number of errors to ignore before throwing an exception
        trace_level (int): verbosity of output (``0``=only errors, ``1``=errors + warning, ``2``=all output)
        chunk_size_in_bytes (int): number of bytes to read from disk in a single read operation (default 32MB)
        randomizationWindow (int): (optional) randomization window in samples. If randomize is set to ``True``, shuffle input data within the specified window size
        keepDataInMemory (bool): (optional) if ``True``, will cache the whole dataset in memory
        frameMode (bool): (optional) if ``True``, will use a packing method optimized for frames (single sample sequences)
    """

    def __init__(self, 
            filename, 
            randomize=True,
            skip_sequence_ids=False,
            max_errors=0,
            trace_level=1,
            chunk_size_in_bytes=32*1024**2,
            randomizationWindow=None,
            keepDataInMemory=False,
            frameMode=False
            ):
        self.reader_type = 'CNTKTextFormatReader'
        self.filename = filename
        self.randomize = bool(randomize)
        self.skip_sequence_ids = bool(skip_sequence_ids)
        self.max_errors = int(max_errors)
        if not self.max_errors >= 0:
            raise ValueError('parameter "max_errors" has to be an integer ' +
                    'greater than or equal to 0. You gave: %s'%max_errors)
        self.trace_level = int(trace_level)
        if not self.trace_level in [0,1,2]:
            raise ValueError('parameter "trace_level" has to be an integer ' +
                    'from [0, 1, 2]. You gave: %s'%str(trace_level))

        self.chunk_size_in_bytes = int(chunk_size_in_bytes)
        if not self.chunk_size_in_bytes > 0:
            raise ValueError('parameter "chunk_size_in_bytes" has to be an integer ' +
                    'greater than zero. You gave: %s'%str(chunk_size_in_bytes))
        
        if self.chunk_size_in_bytes < 0:
            raise ValueError('parameter "chunk_size_in_bytes" has to be an integer ' +
                    'greater than or equal to zero. You gave: %s'\
                            %str(self.chunk_size_in_bytes))
        
        self.randomizationWindow = None
        if (randomizationWindow is not None):
            self.randomizationWindow = int(randomizationWindow)
            if self.randomizationWindow <= 0:
                raise ValueError('parameter "randomizationWindow" has to be an integer ' +
                    'greater than zero. You gave: %s'\
                            %str(self.randomizationWindow))

        self.keepDataInMemory = bool(keepDataInMemory)
        self.frameMode = bool(frameMode)

    def map(self, node_or_name, **kw):
        '''
        Create a mapping from a :class:`cntk.graph.ComputationNode` or a node's name in the
        configuration file to a parameter dictionary. 

        Args:
            node_or_name (:class:`cntk.graph.ComputationNode` or str): node or its variable name
            kw (dict): currently supported parameters are ``alias``, ``dim`` (number of dimensions), and ``format`` (``dense`` or ``sparse``)

        Returns:
            :class:`cntk.reader.InputMap` 
        '''

        return InputMap(self).map(node_or_name, **kw)

    def _to_config_description(self, input_map):
        '''
        Write the reader configuration. For this, all previously registered
        :class:`cntk.reader.LazyInputReader`'s will be serialized into one common file.

        Args:
            input_map (:class:`cntk.reader.InputMap`): describes how to map inputs to the data in a data file using a reader

        Returns:
            string representation of the reader configuration
        '''

        if input_map.is_empty():
            return ''

        if input_map.has_mapped() and input_map.has_unmapped():
            raise ValueError('it is not supported to have a reader together' +
                    ' with inputs that are initialized without a reader' +
                    ' (e.g. NumPy).')

        if input_map.reader is not None and self is not input_map.reader:
            raise ValueError('reader mismatch')

        from cntk.context import get_context
        configuration = {
                'readerType': self.reader_type,
                'file': self.filename,
                'randomize': str(self.randomize).lower(),
                'skipSequenceIds': str(self.skip_sequence_ids).lower(),
                'maxErrors': self.max_errors,
                'traceLevel': self.trace_level,
                'chunkSizeInBytes': self.chunk_size_in_bytes,
                'keepDataInMemory': str(self.keepDataInMemory).lower(),
                'frameMode': str(self.frameMode).lower()
                }

        template = ''' 
        reader = [
            readerType = %(readerType)s
            file = "%(file)s"                
            randomize = %(randomize)s
            skipSequenceIds = %(skipSequenceIds)s
            maxErrors = %(maxErrors)s
            traceLevel = %(traceLevel)i
            chunkSizeInBytes = %(chunkSizeInBytes)i
            keepDataInMemory = %(keepDataInMemory)s
            frameMode = %(frameMode)s
        ''' % configuration

        if (self.randomizationWindow is not None):
            template += '''
            randomizationWindow = %i
        ''' % self.randomizationWindow

        template += '''
            input = [
        '''

        if input_map.has_unmapped():
            if len(input_map.node_map) > 0:
                raise ValueError('you cannot have inputs initialized with '+
                        'NumPy arrays together with inputs that are ' +
                        ' initialized with a custom reader')

            input_map._serialize_unmapped_nodes(
                input_map.unmapped_nodes, self.filename)

        for node_or_name, param_dict in input_map.node_map.items():
            if (isinstance(node_or_name, ComputationNode)):
                name = node_or_name.name
            else:
                name = node_or_name

            if not 'format' in param_dict:
                param_dict['format'] = 'dense'
            
            if not 'dim' in param_dict:
                if isinstance(node_or_name.reader, _LazyInputReaderBase):
                    lazy = node_or_name.reader
                    param_dict.update(node_or_name.reader.param_dict)
                else:
                    raise ValueError('parameter "dim" not specified for node "%s"'%str(node_or_name))

            indent =5*MODEL_INDENTATION*' '
            params = ['%s%s = %s'%(indent, k,v) for k,v in
                    param_dict.items()]

            param_lines = '\n'.join(params)

            if 'alias' in param_dict:
                a = param_dict['alias']
            else:
                a = name

            template += '''
            {0}=[
                {1}
            ]'''.format(name, param_lines)

        template += '''
            ]
        ]
            '''

        return template


class _LazyInputReaderBase(object):

    '''
    Base class of lazy readers that serializes the data to disk only when
    the complete graph is specified. This is necessary in case of multiple
    inputs, because they have to reside in the same file.

    Note:
        All readers of this type need to have the exact same number of samples,
        as they will be aligned by the first index.

    Note:
        This class will be deprecated once the reader bundlers have arrived in
        CNTK.

    Args:
        node (`_InputComputationNodeBase`): node to which this lazy reader is connected
        input_alias (str): a short name for the input, it is how inputs are referenced in the data files. If not provided, it will be automatically assigned.
        dynamic_axis (str or output of :func:`cntk.ops.dynamic_axis`): the dynamic axis packaged as sequences. If not, it will wrapped again in a sequence of length 1.
    '''

    def __init__(self, node, input_alias=None, dynamic_axis=''):
        if not node._is_input():
            raise ValueError('LazyInputReader needs an input node')

        self.node = node
        self.input_alias = input_alias
        self.dynamic_axis = dynamic_axis

class LazyInputReader(_LazyInputReaderBase):

    '''
    Lazy reader that takes an NumPy array and serializes it to disk only when
    the complete graph is specified. This is necessary in case of multiple
    inputs, because they have to reside in the same file.

    Note:
        All readers of this type need to have the exact same number of samples,
        as they will be aligned by the first index.

    Note:
        This class will be deprecated once the reader bundlers have arrived in
        CNTK.

    Args:
        batch (ndarray): the data to be serialized.
        node (`_InputComputationNodeBase`): node to which this lazy reader is connected
        input_alias (str): a short name for the input, it is how inputs are referenced in the data files. If not provided, it will be automatically assigned.
        dynamic_axis (str or output of :func:`cntk.ops.dynamic_axis`): the dynamic axis packaged as sequences. If not, it will wrapped again in a sequence of length 1.
    '''

    def __init__(self, batch, node, input_alias=None, dynamic_axis=''):
        super(LazyInputReader, self).__init__(node, input_alias, dynamic_axis)

        if batch is None:
            raise ValueError(
                'you initalized LazyInputReader without valid batch data')

        self.batch = batch

        shapes_in_tensor = set()
        # make sure that modulo dynamic axis all tensors of one lazy input have
        # the same shape
        for tensor in self.batch:
            if isinstance(tensor, list):
                tensor = np.asarray(tensor)

            if self.dynamic_axis:
                # collecting the shapes ignoring the dynamic axis
                shapes_in_tensor.add(tensor.shape[1:])
            else:
                shapes_in_tensor.add(tensor.shape)

        # ignoring the dynamic axis, all shapes should be equal
        if len(shapes_in_tensor) != 1:
            raise ValueError('except for the sequence dimensions all shapes ' +
                             'should be the same - instead we %s' %
                             (", ".join(str(s) for s in shapes_in_tensor)))

        shape = shapes_in_tensor.pop()
        if not shape:
            shape = (1,)
        
        # cntk uses column major, thus we reverse the shape
        self.shape = self.node.shape = tuple(reversed(shape))

        self.param_dict = {}
        self.param_dict['dim'] = np.multiply.reduce(self.shape)
        self.param_dict['format'] = 'dense'

    def batch_size(self):
        return len(self.batch)

    def data_of_sample(self, idx):
        data = self.batch[idx]
        if not self.dynamic_axis:
            data = np.asarray([data])

        return data

class LazySparseInputReader(_LazyInputReaderBase):

    '''
    Lazy reader that takes an NumPy array and serializes it to disk only when
    the complete graph is specified. This is necessary in case of multiple
    inputs, because they have to reside in the same file.

    Note:
        All readers of this type need to have the exact same number of samples,
        as they will be aligned by the first index.

    Note:
        This class will be deprecated once the reader bundlers have arrived in
        CNTK.

    Args:
        indices (list): list of indices
        values (list): list of values corresponding to indices
        shape (tuple): shape of the input
        node (`_InputComputationNodeBase`): node to which this lazy reader is connected
        input_alias (str): a short name for the input, it is how inputs are referenced in the data files. If not provided, it will be automatically assigned.
        dynamic_axis (str or output of :func:`cntk.ops.dynamic_axis`): the dynamic axis packaged as sequences. If not, it will wrapped again in a sequence of length 1.
    '''

    def __init__(self, indices, values, shape, node, input_alias=None, dynamic_axis=''):
        super(LazySparseInputReader, self).__init__(node, input_alias, dynamic_axis)

        if not indices or not values or not shape:
            raise ValueError(
                'you initalized LazySparseInputReader without valid initialization')

        if not len(indices) == len(values):
            raise ValueError('indices has different length than values')

        self.indices = indices
        self.values = values

        # cntk uses column major, thus we reverse the shape
        self.shape = self.node.shape = tuple(reversed(shape))


        self.param_dict = {}
        self.param_dict['dim'] = np.multiply.reduce(self.shape)
        self.param_dict['format'] = 'sparse'

    def batch_size(self):
        return len(self.indices)

    def data_of_sample(self, idx):
        indices = self.indices[idx]
        values = self.values[idx]

        data = dict(zip(indices,values))

        if not self.dynamic_axis:
            data = [data]

        return data

class AbstractReaderAggregator(with_metaclass(ABCMeta, dict)):

    """ This is the abstract reader class. The sub-classes of this class
    are not exposed to the user and are used to aggregate all inputs' readers
    for a graph before generating the CNTK config. That is, they are a mirror
    to what we see under the reader block in CNTK config files.
    """

    @abstractmethod
    def _to_config_description(self):
        """Generate the reader configuration block
        """
        raise NotImplementedError

    # required so that instances can be put into a set
    def __hash__(self): return hash(id(self))

    def __eq__(self, x): return x is self

    def __ne__(self, x): return x is not self


class UCIFastReaderAggregator(AbstractReaderAggregator):

    """This is the reader class the maps to UCIFastReader of CNTK

    Args:
        filename (str): data file path
        custom_delimiter (str): the default is space and tab, you can specify other delimiters to be used        
    """

    def __init__(self, filename, custom_delimiter=None):
        """ Reader constructor    
        """
        self["ReaderType"] = "UCIFastReader"
        self["FileName"] = filename
        self["CustomDelimiter"] = custom_delimiter
        self.inputs_def = []

    def add_input(self, node_or_name, input_start, input_dim, num_of_classes=None, label_mapping_file=None):
        """Add an input to the reader

        Args:
            node_or_name (str or ComputationNode): either name of the input in the network definition or the node itself
            input_start (int): the start column   
            input_dim (int): the number of columns
            num_of_classes (int): the number of classes
            label_mapping_file (str): the mapping file path, it can be simply with all the possible classes, one per line
        """
        if not node_or_name or input_start is None or input_dim is None:
            raise ValueError("one of the parameters of add_input is None")

        self.inputs_def.append(
            (node_or_name, input_start, input_dim, num_of_classes, label_mapping_file))

    def _to_config_description(self):
        """Generate the reader configuration block
        """
        template = '''\
    reader = [
        traceLevel = 2
        readerType = "%(ReaderType)s"
        file = "%(FileName)s"
        randomize = "none"
        verbosity = 1
'''

        if self['CustomDelimiter'] is not None:
            template += '''\
        customDelimiter = %(CustomDelimiter)s
       '''

        if self.inputs_def is not None:
            for (node_or_name, start, dim, num_of_classes, map_file) in self.inputs_def:
                if (isinstance(node_or_name, ComputationNode)):
                    name = node_or_name.name
                else:
                    name = node_or_name

                template += '''
        {0} = [
            start = {1}
            dim = {2}
            '''.format(name, start, dim)

                if num_of_classes:
                    template += '''\
            labelDim= {0}
                '''.format(num_of_classes)
                if map_file:
                    template += '''\
            labelMappingFile= "{0}"
                '''.format(map_file)

                template += '''
        ]
'''

            template += '''\
    ]            
'''

        return template % self

class InputMap(object):
    '''
    An instance of `InputMap` is return by the readers' `.map()` function, and
    binds input nodes to the aliases in a reader.

    Args:
        reader (:class:`cntk.reader.CNTKTextFormatReader`): the reader for which this instance defines the mapping. If ``None``, then the inputs are expected to be NumPy arrays, and an temporary reader will be set up automatically.

    Example::
    
       train_reader = CNTKTextFormatReader('file.txt')
       with ctx.train(..., input_map=train_reader.map(X, shape='I').map(y, shape='L')):
            ...
    '''
    def __init__(self, reader=None):
        self.reader = reader
        self.node_map = {}
        self.unmapped_nodes = set()

    def __contains__(self, node_or_name):
        if node_or_name in self.node_map:
            return True

        if isinstance(node_or_name, ComputationNode):
            if node_or_name.name and \
                    node_or_name.name in self.node_map:
                return True

        return False


    def map(self, node_or_name, **kw):
        '''
        Updates the input map by the additional mapping from `node_or_name`
        to the parameter settings in `kw`.

        Args:
            node_or_name (:class:`cntk.graph.ComputationNode` or str): node or its variable name
            kw (dict): currently supported parameters are ``alias``, ``dim`` (number of dimensions), and ``format`` (``dense`` or ``sparse``)

        Returns:
            :class:`cntk.reader.InputMap`, such that you can chain multiple invocations of `map()`.
        '''
        self.node_map[node_or_name] = kw
        return self

    def has_mapped(self):
        return len(self.node_map)>0

    def has_unmapped(self):
        return len(self.unmapped_nodes)>0

    def is_empty(self):
        return not self.has_mapped() and not self.has_unmapped()

    def _to_config_description(self, directory=None):
        if self.reader is None:
            if not self.unmapped_nodes:
                # No inputs in the graph
                return ''

            # We have found only inputs that were directly initialized.
            # In this case, we need to serialize them into one file.

            from .utils import get_temp_filename

            filename = get_temp_filename(directory)
            if len(self.node_map) > 0:
                raise ValueError('you cannot have inputs initialized with '+
                        'NumPy arrays together with inputs that are ' +
                        ' initialized with a custom reader')

            self._serialize_unmapped_nodes(filename)
            
            # All the data we got, was through NumPy. In this case, we assume
            # that all the required randomization has happened already.
            r = CNTKTextFormatReader(filename, randomize=None)

            return r._to_config_description(self)

        else:
            return self.reader._to_config_description(self)

    def _add_unmapped(self, node):
        '''
        In case node had been initialized directly with a tensor, it is
        accumulated. At the end, all of these nodes will be serialized into one
        temporary file.
        '''
        self.unmapped_nodes.add(node)

    def _serialize_unmapped_nodes(self, filename):
        '''
        Generates a file readable with `cntk.readers.CNTKTextFormatReader` that
        can be connected to the inputs of the network and fills in missing
        information in the lazy input defs (shape, alias).

        Args:
            filename (str): name of the file
        '''

        alias_counter = 0
        sample_sizes = collections.defaultdict(list)
        used_aliases = set()
        for node in self.unmapped_nodes:
            is_lazy_input = isinstance(node.reader, _LazyInputReaderBase)
            if not (node._is_input() and is_lazy_input):
                raise ValueError('expected NumPy input, but got "%s"'%str(node))
            
            l = node.reader

            # make sure all inputs have valid unique aliases
            if l.input_alias is None or l.input_alias.startswith('_'):
                new_alias = '_I_%i' % alias_counter
                alias_counter += 1
                while new_alias in used_aliases:
                    new_alias = '_I_%i' % alias_counter
                    alias_counter += 1

                l.input_alias = new_alias
                used_aliases.add(new_alias)

            # keep track of sample sizes
            sample_sizes[l.batch_size()].append(l.input_alias)


            self.node_map[node] = { 'alias': l.input_alias }

        # make sure all inputs have same sample size
        if len(sample_sizes) != 1:
            raise ValueError(
                'LazyInputReaders have different sizes: %s' % str(sample_sizes))

        sample_size = list(sample_sizes)[0]

        # ready to serialize
        with open(filename, 'w') as f:
            for idx in range(sample_size):
                alias_tensor_map = {}
                for node in self.unmapped_nodes:
                    l = node.reader
                    alias_tensor_map[l.input_alias] = l.data_of_sample(idx)
                f.write(tensors_to_text_format(idx, alias_tensor_map) + '\n')

        self.unmapped_nodes.clear()
