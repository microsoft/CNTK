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

    """Abstract class that represents a reader for one input node.
    """

    # required so that instances can be put into a set
    def __hash__(self): return hash(id(self))

    def __eq__(self, x): return x is self

    def __ne__(self, x): return x is not self

    def _to_aggregate_form():
        pass


class UCIFastReader(AbstractReader):

    """`Deprecated` - A UCIFastReader for one input node. Please switch to
    :class:`CNTKTextFormatReader`.

    Note that the dimensions are not inferred from the input node's shape,
    because in case of a label node the dimension does not match the shape
    which would be (``numOfClasses``,1).

    Args:
        filename (str): the name of the file where the data is stored
        custom_delimiter (str): what delimiter is used to separate columns, specify
        it in case it neither tab nor white spaces.
        input_start (int): the start column   
        input_dim (int): the number of columns
        num_of_classes (int): the number of classes
        label_mapping_file (str): the mapping file path, it can be simply with
        all the possible classes, one per line
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

    """A CNTKTextFormatReader for one input node that supports sequences. 

    Args:
        filename (str): the name of the file where the data is stored
        input_alias (str): a short name for the input, it is how inputs are referenced in the data files        
        format (str): 'dense' or 'sparse'

    Example:
       The following example encodes two samples, one has a sequence of one
       scalar, while the second has a sequence of two scalars::

           0\t|I 60.0
           1\t|I 22.0
           1\t|I 24.0

       The ``I`` is the alias, which would be used to connect the data to the
       input node. Let's say the above data is stored in ``data.txt``, you would
       set up the reader as follows::

           r = CNTKTextFormatReader('data.txt', 'I')


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

       The normal matrix based format, for which you would have used
       :class:`UCIFastReader` in the past can be simply converted by prepending
       every line by the line number and a bar (``|``). Of course it only works
       for sequences of length 1, since in matrix format you cannot go beyond
       that:

       :class:`UCIFastReader` format::

           0 1
           10 11
           20 21

       can be easily converted to the :class:`CNTKTextFormatReader` format
       (using alias `I`)::

           0\t|I 0 1
           1\t|I 10 21
           2\t|I 20 21
    """

    def __init__(self, filename):
        self.filename = filename

    def map(self, node_or_name, **kw):
        '''
        Create a mapping from a `ComputationNode` or a node's name in the
        configuration file to a parameter dictionary. Parameters:

        Args:
            node_or_name (`ComputationNode` or str): node or its variable name
            alias (str): the alias in the data file. If omitted, the node's variable
        name will be taken.
            dim (int): the dimension of the imput
        '''

        return InputMap(self).map(node_or_name, **kw)

    def generate_config(self, input_map):
        '''
        Write the reader configuration. For this, all previously registered
        `LazyInputReader`s will be serialized into one common file.

        Args:
            input_map (`InputMap`): describes how to map inputs to the data in a data file using a reader

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
        template = ''' 
        reader = [
            traceLevel = 2
            readerType = CNTKTextFormatReader
            file = "%(FileName)s"                
        ''' % {'FileName': self.filename}

        template += '''
            input = [
        '''

        if input_map.has_unmapped():
            input_map._serialize_unmapped_nodes(
                input_map.unmapped_nodes, self.filename)


        for node_or_name, param_dict in input_map.node_map.items():
            if (isinstance(node_or_name, ComputationNode)):
                name = node_or_name.var_name
            else:
                name = node_or_name

            if not 'format' in param_dict:
                param_dict['format'] = 'dense'
            
            if not 'dim' in param_dict:
                if isinstance(node_or_name.reader, LazyInputReader):
                    lazy = node_or_name.reader
                    param_dict['dim'] = np.multiply.reduce(lazy.shape)
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


class LazyInputReader(object):

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
        node (`InputComputationNodeBase`): node to which this lazy reader is
        connected
        input_alias (str): a short name for the input, it is how inputs are
        referenced in the data files. If not provided, it will be automatically
        assigned.
        has_dynamic_axis (bool): whether the tensor has already the data
        packaged as sequences. If not, it will wrapped again in a sequence of
        length 1.
    '''

    def __init__(self, batch, node, input_alias=None, has_dynamic_axis=True):
        if not batch:
            raise ValueError(
                'you initalized LazyInputReader without valid batch data')

        self.batch = batch
        if not node.is_input():
            raise ValueError('LazyInputReader needs an input node')

        self.node = node

        sample = batch[0]
        if has_dynamic_axis:
            # collecting the shapes ignoring the dynamic axis
            self.node.dims = np.asarray(sample).shape[1:]
        else:
            self.node.dims = np.asarray(sample).shape

        self.input_alias = input_alias
        self.has_dynamic_axis = has_dynamic_axis


class AbstractReaderAggregator(with_metaclass(ABCMeta, dict)):

    """ This is the abstract reader class. The sub-classes of this class
    are not exposed to the user and are used to aggregate all inputs' readers
    for a graph before generating the CNTK config. That is, they are a mirror
    to what we see under the reader block in CNTK config files.
    """

    @abstractmethod
    def generate_config(self):
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

    def generate_config(self):
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
                    name = node_or_name.var_name
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
        reader (descendent of `AbstractReader`)
    Example:
        
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
            if node_or_name.var_name and \
                    node_or_name.var_name in self.node_map:
                return True

        return False


    def map(self, node_or_name, **kw):
        self.node_map[node_or_name] = kw
        return self

    def has_mapped(self):
        return len(self.node_map)>0

    def has_unmapped(self):
        return len(self.unmapped_nodes)>0

    def is_empty(self):
        return not self.has_mapped() and not self.has_unmapped()

    def generate_config(self):
        if self.reader is None:
            # We have found only inputs that were directly initialized.
            # In this case, we need to serialize them into one file.

            from .context import get_context
            from .utils import get_temp_filename
            filename = get_temp_filename(get_context().directory)

            assert not self.node_map
            self._serialize_unmapped_nodes(filename)
            
            r = CNTKTextFormatReader(filename)

            return r.generate_config(self)

        else:
            return self.reader.generate_config(self)

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
            assert node.is_input()
            assert isinstance(node.reader, LazyInputReader)
            
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
            sample_sizes[len(l.batch)].append(l.input_alias)

            shapes_in_tensor = set()

            # make sure that modulo dynamic axis all tensors of one lazy input have
            # the same shape
            for tensor in l.batch:
                if isinstance(tensor, list):
                    tensor = np.asarray(tensor)

                if l.has_dynamic_axis:
                    # collecting the shapes ignoring the dynamic axis
                    shapes_in_tensor.add(tensor.shape[1:])
                else:
                    shapes_in_tensor.add(tensor.shape)

            # ignoring the dynamic axis, all shapes should be equal
            if len(shapes_in_tensor) != 1:
                raise ValueError('except for the sequence dimensions all shapes ' +
                                 'should be the same - instead we %s' %
                                 (", ".join(str(s) for s in shapes_in_tensor)))

            # shapes_in_tensor now contains only one shape, which has the sequence
            # dimension removed.
            value_shape = shapes_in_tensor.pop()
            l.shape = value_shape if value_shape else (1,)

            assert node not in self.node_map
            self.node_map[node] = {
                    'alias': l.input_alias, 
                    }

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
                    if l.has_dynamic_axis:
                        alias_tensor_map[l.input_alias] = l.batch[idx]
                    else:
                        alias_tensor_map[l.input_alias] = [l.batch[idx]]
                f.write(tensors_to_text_format(idx, alias_tensor_map) + '\n')

        self.unmapped_nodes.clear()
