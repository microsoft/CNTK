# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from .. import cntk_py

MAX_UI64 = int('0xffffffffffffffff', 16)


class MinibatchSource(cntk_py.MinibatchSource):
    '''
    Parent class of all minibatch sources. For most cases you will need the
    helper functions `:func:cntk.io.text_format_minibatch_source` or
    `:func:cntk.io.create_minibatch_source`.
    '''

    def stream_infos(self):
        '''
        Describes the stream that this source produces.
        '''
        return super(MinibatchSource, self).stream_infos()

    def stream_info(self, name):
        '''
        Gets the description of the stream with given name. 
        Throws an exception of there are none or multiple streams with this
        same name.
        '''
        return super(MinibatchSource, self).stream_info(name)
        
    def get_next_minibatch(self, minibatch_size_in_samples, device = None):
        '''
        Reads a minibatch that contains data for all input streams.
        The minibatch size is specified terms of #samples and/or #sequences for the primary input stream; value of 0 for #samples/#sequences means unspecified.
        In case the size is specified in terms of both #sequences and #samples, the smaller of the 2 is taken.
        An empty map is returned when the MinibatchSource has no more data to return.''

        Args:
            minibatch_size_in_samples (int): number of samples to retrieve for
             the next minibatch. Must be > 0.
        '''
        if device is None:
            device = cntk_py.DeviceDescriptor.use_default_device()

        return super(MinibatchSource, self).get_next_minibatch(\
                minibatch_size_in_samples,
                minibatch_size_in_sequences, device)


def _py_dict_to_cntk_dict(py_dict):
    '''
    Converts a Python dictionary into a CNTK Dictionary whose values are CNTK DictionaryValue instances.
    Args:
        py_dict (dict): a dictionary to be converted.
    Returns: 
        :class:`cntk_py.Dictionary`
    '''
    res = cntk_py.Dictionary();
    for k,v in py_dict.items():
        if isinstance(v,dict):
            res[k] = cntk_py.DictionaryValueFromDict(_py_dict_to_cntk_dict(v))
        #TODO: add support to list of lists ?
        elif isinstance(v,list):
            l = list()
            for e in v:
                if isinstance(e,dict):
                    l.append(cntk_py.DictionaryValueFromDict(_py_dict_to_cntk_dict(e)))
                else:
                    l.append(cntk_py.DictionaryValue(v))
            res[k] = cntk_py.DictionaryValue(l)
        else:
            res[k] = cntk_py.DictionaryValue(v)
    return res
        
def minibatch_source(config_dict):
    '''
    Instantiate the CNTK built-in composite minibatch source which is used to stream data into the network.    
    Args:
        config_dict (dict): a dictionary containing all the key-value configuration entries.
    Returns: 
        :class:`cntk_py.MinibatchSource`
    '''
    cntk_dict = _py_dict_to_cntk_dict(config_dict)
    return cntk_py.create_composite_minibatch_source(cntk_dict)

    
def composite_minibatch_source(configuration):
    '''
    Instantiate the CNTK built-in composite minibatch source.

    Args:
        configuration (`dict`): dictionary of (potential dictionaries of)
         source configurations

    Returns:
        `:class:cntk.io.MinibatchSource`
    '''
    return cntk_py.create_composite_minibatch_source(configuration)

#
# CNTKTextFormatReader
#

def text_format_minibatch_source(path, stream_configs, epoch_size=MAX_UI64):
    '''
    Creates a minibatch source from a CNTKTextFormatReader file.

    Args:
        path ('file'): filename of the data file
        stream_configs (`list` of `:class:StreamConfiguration` instances): list
         of stream configurations, each of which describes one stream in the
         file
        epoch_size (`int`, optional): size of an epoch. In case of 0 the size
         of the training set will be taken. Default is max of 64bit.

    Returns:
        `:class:cntk.io.MinibatchSource'
    '''
    return cntk_py.text_format_minibatch_source(path, stream_configs,
            epoch_size)

class StreamConfiguration(cntk_py.StreamConfiguration):
    '''
    Configuration of a stream in a text format reader. This can be used
    `:func:cntk.io.text_format_minibatch_source`.

    Args:
        name (`str`): name of this stream
        dim (`int`): dimensions of this stream. A text format reader reads data
        as flat arrays. If you need different shapes you can
        `:func:cntk.ops.reshape` it later.
        is_sparse (`bool`, optional): whether the provided data is sparse
         (`False` by default)
        stream_alias (`str`): name of the stream in the file that is fed to the
         `:func:cntk.io.text_format_minibatch_source`       
    '''
    def __init__(self, name, dim, is_sparse=False, stream_alias=''):
        return super(StreamConfiguration, self).__init__(name, dim, is_sparse, stream_alias)

