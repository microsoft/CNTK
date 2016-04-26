# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from abc import ABCMeta, abstractmethod
import numpy as np

from .utils import with_metaclass, serialize_input_data
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
        self.inputs_def = []
        self.lazy_inputs_def = []

    def add_input(self, name_or_node, input_start, input_dim, num_of_classes=None, label_mapping_file=None):
        """Add an input to the reader

        Args:
            name_or_node (str or ComputationNode): either name of the input in the network definition or the node itself
            input_start (int): the start column   
            input_dim (int): the number of columns
            num_of_classes (int): the number of classes
            label_mapping_file (str): the mapping file path, it can be simply with all the possible classes, one per line
        """
        if not name_or_node or input_start is None or input_dim is None:
            raise ValueError("one of the parameters of add_input is None")

        self.inputs_def.append(
            (name_or_node, input_start, input_dim, num_of_classes, label_mapping_file))

    def add_lazy_input(self, lazy):
        '''
        Add a LazyInputReader instance. Lazy inputs will be ignore in case
        normal readers are defined. Thereby, we make sure that we can train
        using lazy inputs, and later infer using different test data.

        Args:
            lazy (LazyInputReader): contains data that will be merged later
            with other inputs
        '''
        self.lazy_inputs_def.append(lazy)

    def generate_config(self):
        '''
        Write the reader configuration. For this, all previously registered
        `LazyInputReader`s will be serialized into one common file.
        '''

        from cntk.context import get_context
        template = ''' 
        reader = [
            traceLevel = 2
            readerType = CNTKTextFormatReader
            file = "%(FileName)s"                
        ''' % {'FileName': self.filename}

        if self.inputs_def or self.lazy_inputs_def:
            template += '''
                input = [
            '''

        if self.inputs_def:
            for (name_or_node, input_alias, dim, format) in self.inputs_def:
                if (isinstance(name_or_node, ComputationNode)):
                    name = name_or_node.var_name
                else:
                    name = name_or_node

                if not input_alias:
                    a = name
                else:
                    a = input_alias

                template += '''
                {0}=[
                    alias = "{1}"                
                    dim = {2}          
                    format = "{3}"
                ]'''.format(name, a, dim, format)

        if not self.inputs_def and self.lazy_inputs_def:
            alias_lazy_map = serialize_input_data(
                self.lazy_inputs_def, self.filename)

            for alias, lazy in alias_lazy_map.items():

                dim = np.multiply.reduce(lazy.shape)
                format = 'dense'  # TODO: sparse

                template += '''
                {0}=[
                    alias = "{1}"                
                    dim = {2}          
                    format = "{3}"
                ]'''.format(lazy.node.var_name, alias, dim, format)

        if self.inputs_def or self.lazy_inputs_def:
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

    def add_input(self, name_or_node, input_start, input_dim, num_of_classes=None, label_mapping_file=None):
        """Add an input to the reader

        Args:
            name_or_node (str or ComputationNode): either name of the input in the network definition or the node itself
            input_start (int): the start column   
            input_dim (int): the number of columns
            num_of_classes (int): the number of classes
            label_mapping_file (str): the mapping file path, it can be simply with all the possible classes, one per line
        """
        if not name_or_node or input_start is None or input_dim is None:
            raise ValueError("one of the parameters of add_input is None")

        self.inputs_def.append(
            (name_or_node, input_start, input_dim, num_of_classes, label_mapping_file))

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
            for (name_or_node, start, dim, num_of_classes, map_file) in self.inputs_def:
                if (isinstance(name_or_node, ComputationNode)):
                    name = name_or_node.var_name
                else:
                    name = name_or_node

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
