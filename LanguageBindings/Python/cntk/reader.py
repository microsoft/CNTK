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
    
    :param file: data file path
    :param features_dim: number of feature columns
    :param labels_dim: number of label columns
    :param features_start: the index of the first feature column
    :param labels_start: the index of the first label column
    :param num_of_classes: the number of classes
    :param label_mapping_file: 
        the mapping file path, it can be simply with all the possible classes, one per line
    :param custom_delimiter: the default is space and tab, you can specify other delimiters to be used
    """
    
    def __init__(self, file, features_dim, labels_dim, features_start, labels_start, \
                num_of_classes, label_mapping_file, custom_delimiter = None):
        """ Reader constructor    
        """    
            
        self["readerType"] = self.__class__.__name__
        self["file"] = file 
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
        		file = "%(file)s"
        		randomize = "none"
        		verbosity = 1          
               %(customDelimiter)s
               
        		features=[
        			start = "%(featuresStart)s"
        			dim = "%(featuresDim)s"
        		]
        	
        		labels=[
        			start = "%(labelsStart)s"
        			dim = "%(labelsDim)s"		          
                    labelDim="%(numOfClasses)s"        			
                    labelMappingFile="%(labelMappingFile)s" 
        		]
        '''
                            
        config = template%self
        return config 
    