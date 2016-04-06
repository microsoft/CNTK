from abc import ABCMeta, abstractmethod
import numpy as np

from .graph import ComputationNode


class AbstractReader(dict, metaclass=ABCMeta):

    """ This is the abstract reader class
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


class UCIFastReader(AbstractReader):

    """This is the reader class the maps to UCIFastReader of CNTK
    :param filename: data file path
    :param custom_delimiter: the default is space and tab, you can specify other delimiters to be used        
    """

    def __init__(self, filename, custom_delimiter=None):
        """ Reader constructor    
        """
        self["ReaderType"] = self.__class__.__name__
        self["FileName"] = filename
        self["CustomDelimiter"] = custom_delimiter
        self.inputs_def = []

    def add_input(self, name_or_node, input_start, input_dim, num_of_classes=None, label_mapping_file=None):
        """Add an input to the reader
        :param name_or_node: either name of the input in the network definition or the node itself
        :param input_start: the start column   
        :param input_dim: the number of columns
        :param num_of_classes: the number of classes
        :param label_mapping_file: 
            the mapping file path, it can be simply with all the possible classes, one per line
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


class CNTKTextFormatReader(AbstractReader):

    """This is the reader class the maps to CNTKTextFormatReader of CNTK
    :param filename: data file path
    """

    def __init__(self, filename):
        """ Reader constructor    
        """
        self["ReaderType"] = self.__class__.__name__
        self["FileName"] = filename
        self.inputs_def = []

    def add_input(self, name_or_node, input_alias, input_dim, format="Dense"):
        """Add an input to the reader
        name_or_node: the name of the input in the network definition or the node itself
        input_alias: a short name for the input, it is how inputs are referenced in the data files
        input_dim: the lenght of the input vector
        format: dense or sparse
        """
        if not name_or_node or input_dim is None or format is None:
            raise ValueError("one of the parameters of add_input is None")

        input_alias = input_alias or name_or_node

        self.inputs_def.append((name_or_node, input_alias, input_dim, format))

    def generate_config(self):
        """Generate the reader configuration block
        """
        template = ''' 
        reader = [
            traceLevel = 2
            readerType = "%(ReaderType)s"
            file = "%(FileName)s"                
        '''

        if self.inputs_def is not None:
            template += '''
                input = [
            '''

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

            template += '''
            ]
        ]
            '''
        return template % self


def NumPyReader(data, filename):
    # TODO: get rid of this
    """
    This is a factory that wraps Python arrays with a UCIFastReader.
    """

    data = np.asarray(data)
    if len(data.shape) == 1:
        num_cols = 1
    elif len(data.shape) == 2:
        num_cols = data.shape[1]
    else:
        raise ValueError('NumPyReader does not support >2 dimensions')

    format_str = ' '.join(['%f'] * num_cols)
    np.savetxt(filename, data, delimiter=' ', newline='\r\n', fmt=format_str)

    return UCIFastReader(filename)
