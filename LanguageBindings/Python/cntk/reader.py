from abc import ABCMeta, abstractmethod

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
    :param features_dim: number of label columns
    :param features_start: the index of the first label column    
    :param labels_dim: number of label columns
    :param labels_start: the index of the first label column
    :param num_of_classes: the number of classes
    :param label_mapping_file: 
        the mapping file path, it can be simply with all the possible classes, one per line
    :param custom_delimiter: the default is space and tab, you can specify other delimiters to be used
    """
    
    def __init__(self, filename, \
            features_dim = None, \
            features_start= None, \
            labels_dim=None, \
            labels_start=None, \
            num_of_classes=None, \
            label_mapping_file=None, \
            custom_delimiter = None):
        """ Reader constructor    
        """                
        self["ReaderType"] = self.__class__.__name__
        self["FileName"] = filename
        self["LabelsDim"] = labels_dim
        self["LabelsStart"] = labels_start
        self["FeaturesDim"] = features_dim
        self["FeaturesStart"] = features_start
        self["NumOfClasses"] = num_of_classes          
        self["LabelMappingFile"] = label_mapping_file
        self["CustomDelimiter"] = custom_delimiter

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
        #TODO: generalize the reader to take n features sequences and m label sequences
        if self['FeaturesStart'] is not None:
            template += '''

        		features=[
        			start = "%(FeaturesStart)s"
        			dim = "%(FeaturesDim)s"		          
        		]'''
        
        
        if self['LabelsStart'] is not None:
            template += '''

        		labels=[
        			start = "%(LabelsStart)s"
        			dim = "%(LabelsDim)s"		          
                    labelDim="%(NumOfClasses)s"        			
                    labelMappingFile="%(LabelMappingFile)s" 
        		]'''
                            
        return template%self
    
def NumPyReader(data, filename): 
    """
    This is a convenience function that wraps Python arrays.
    """
    
    import numpy as np
    data = np.asarray(data)
    format_str = ' '.join(['%f']*data.shape[1])
    np.savetxt(filename, data, delimiter=' ', newline='\r\n', fmt=format_str)

    return UCIFastReader(\
            filename, 
            labels_dim=None, \
            labels_start=None, \
            num_of_classes=None, label_mapping_file=None)
