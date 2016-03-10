from abc import ABCMeta, abstractmethod

class AbstractReader(dict, metaclass=ABCMeta):
    """ This is the abstract reader class
    """
    
    @abstractmethod
    def generate_config(self):
        """Generate the reader configuration block
        """
        raise NotImplementedError 

class UCIFastReader(AbstractReader):        
    """This is the reader class
    
    :param filename: data file path
    :param features_dim: number of feature columns
    :param labels_dim: number of label columns
    :param features_start: the index of the first feature column
    :param labels_start: the index of the first label column
    :param num_of_classes: the number of classes
    :param label_mapping_file: 
        the mapping file path, it can be simply with all the possible classes, one per line
    :param custom_delimiter: the default is space and tab, you can specify other delimiters to be used
    """
    
    def __init__(self, filename, \
            features_dim=None, labels_dim=None, \
            features_start=None, labels_start=None, \
            num_of_classes=None, \
            label_mapping_file=None, \
            custom_delimiter = None):
        """ Reader constructor    
        """    
            
        self["readerType"] = self.__class__.__name__
        self["filename"] = filename 
        self["featuresDim"] = features_dim
        self["labelsDim"] = labels_dim
        self["featuresStart"] = features_start
        self["labelsStart"] = labels_start
        self["numOfClasses"] = num_of_classes          
        self["labelMappingFile"] = label_mapping_file
        if (custom_delimiter is not None):
            self["customDelimiter"] = 'customDelimiter = "{0}"'.format(custom_delimiter)
        else:
            self["customDelimiter"] = None

    def generate_config(self):
        """Generate the reader configuration block
        """
        template = '''
        		readerType = "%(readerType)s"
        		file = "%(filename)s"
        		randomize = "none"
        		verbosity = 1          
               '''
        if self['customDelimiter']:
            template += '''
               customDelimiter=%(customDelimiter)s
               '''

        if self['featuresStart']:
            template += '''
               
        		features=[
        			start = "%(featuresStart)s"
        			dim = "%(featuresDim)s"
        		]
                '''
        
        if self['labelsStart']:
            template += '''

        		labels=[
        			start = "%(labelsStart)s"
        			dim = "%(labelsDim)s"		          
                    labelDim="%(numOfClasses)s"        			
                    labelMappingFile="%(labelMappingFile)s" 
        		]
                '''
                            
        return template%self
    
def NumPyReader(data, filename): 
    """
    This is the reader class for bare Python arrays.
    """
    
    import numpy as np
    data = np.asarray(data)
    format_str = ' '.join(['%f']*data.shape[1])
    np.savetxt(filename, data, delimiter=' ', newline='\r\n', fmt=format_str)

    return UCIFastReader(\
            filename, 
            # Features are stored per column, but CNTK transforms them and uses
            # them per row. That's why we take the number of columns here.
            features_dim=data.shape[1], labels_dim=None, \
            features_start=0, labels_start=None, \
            num_of_classes=None, label_mapping_file=None)
