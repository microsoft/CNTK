from abc import ABCMeta, abstractmethod
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

    """This is the reader class
    :param filename: data file path
    :param labels_node_name: the name of the labels node in the network
    :param labels_dim: number of label columns
    :param labels_start: the index of the first label column
    :param num_of_classes: the number of classes
    :param label_mapping_file: 
        the mapping file path, it can be simply with all the possible classes, one per line
    :param custom_delimiter: the default is space and tab, you can specify other delimiters to be used        
    :param inputs_def: is a list of tuples (name_or_node, input_start, input_dim)
        name_or_node: the name of the input in the network definition or the node itself
        input_start: the start column   
        input_dim: the number of columns
    """

    def __init__(self, filename,
                 labels_node_name="labels",
                 labels_dim=None,
                 labels_start=None,
                 num_of_classes=None,                 
                 label_mapping_file=None,                 
                 custom_delimiter=None,
                 inputs_def=None):
        """ Reader constructor    
        """
        self["ReaderType"] = self.__class__.__name__
        self["FileName"] = filename
        self["LabelsNodeName"] = labels_node_name
        self["LabelsDim"] = labels_dim
        self["LabelsStart"] = labels_start            
        self["NumOfClasses"] = num_of_classes
        self["LabelMappingFile"] = label_mapping_file
        self["CustomDelimiter"] = custom_delimiter
        self.inputs_def = inputs_def or []

    def add_input(self, name_or_node, input_start, input_dim):
        """Add an input to the reader
        :param name_or_node: either name of the input in the network definition or the node itself
        :param input_start: the start column   
        :param input_dim: the number of columns
        """
        if (not (name_or_node and input_start and input_dim)):
            raise ValueError("one of the parameters of add_input is None or empty string") 
        
        self.inputs_def.append((name_or_node, input_start, input_dim))
        
    def generate_config(self):
        """Generate the reader configuration block
        """
        template = '''
            readerType = "%(ReaderType)s"
            file = "%(FileName)s"
            randomize = "none"
            verbosity = 1          
            '''

        if self['CustomDelimiter'] is not None:
            template += '''
      customDelimiter=%(CustomDelimiter)s
       '''
               
        if self['LabelsStart'] is not None:
            template += '''
            %(LabelsNodeName)s=[
                start = "%(LabelsStart)s"
                dim = "%(LabelsDim)s"
                labelDim="%(NumOfClasses)s"        			
                labelMappingFile="%(LabelMappingFile)s" 
            ]'''

        if self.inputs_def is not None:
            for (name_or_node, start, dim) in self.inputs_def:                
                if (isinstance(name_or_node, ComputationNode)):
                    name = name_or_node.var_name
                else:
                    name = name_or_node
                template += '''
            {0}=[
                start = {1}
                dim = {2}		          
            ]'''.format(name, start, dim)


        return template % self


def NumPyReader(data, filename):
    """
    This is a convenience function that wraps Python arrays.
    """

    import numpy as np
    data = np.asarray(data)
    format_str = ' '.join(['%f'] * data.shape[1])
    np.savetxt(filename, data, delimiter=' ', newline='\r\n', fmt=format_str)

    return UCIFastReader(
        filename,
        labels_dim=None,
        labels_start=None,
        num_of_classes=None, label_mapping_file=None)
