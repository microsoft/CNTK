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
        self["ReaderType"] = self.__class__.__name__
        self["File"] = file 
        self["FeaturesDim"] = features_dim
        self["LabelsDim"] = labels_dim
        self["FeaturesStart"] = features_start
        self["LabelsStart"] = labels_start
        self["NumOfClasses"] = num_of_classes          
        self["LabelMappingFile"] = label_mapping_file
        if (custom_delimiter is not None):
            self["CustomDelimiter"] = 'customDelimiter = "{0}"'.format(custom_delimiter)
        else:
            self["CustomDelimiter"] = None

    def generate_config(self):
        """Generate the reader configuration block
        """
        template = '''
        		readerType = "%(ReaderType)s"
        		file = "%(File)s"
        		randomize = "none"
        		verbosity = 1          
               %(CustomDelimiter)s
               
        		features=[
        			start = "%(FeaturesStart)s"
        			dim = "%(FeaturesDim)s"
        		]
        	
        		labels=[
        			start = "%(LabelsStart)s"
        			dim = "%(LabelsDim)s"		          
                    labelDim="%(NumOfClasses)s"        			
                    labelMappingFile="%(LabelMappingFile)s" 
        		]'''
                            
        config = template%self
        return config 
    