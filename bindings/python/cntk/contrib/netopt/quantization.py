import cntk as C
from cntk.contrib.netopt.custom_convolution_ops import *


def binarize_convolution(model, train_function, filter_function = None):
    '''
    Replace Convolution layers in the model to Halide implementations of
    binarized convolutions.

    Args:
        model          : model that needs convolutions to be optimized.
        train_function : this is a two step process. First, convert the model 
                         to binary convolution and next, transform it to the 
                         Halide implementation. Between above steps,
                         we need to train the model to get the best results. 
                         train_function provides this functionality.
        filter_function: filter layers in the model to apply the binarization.
                        
    Returns:
        A model with Halid operators.
    '''
    assert(train_function) # needs a training function to convert.    

    z = convert_to_binary_convolution(model)
    train_function(z)
    return convert_to_native_binary_convolution(z)


def convert_to_native_binary_convolution(model):
    '''
    Clones a binary convolution network, sharing the original parameters 
    but substitutes the python 'BinaryConvolution' Function instances 
    used during training with faster C++ NativeBinaryConvolveFunction
    instances that uses optimized binary convolution implementations 
    generated using the Halide framework 

    Args:
        model   : model that needs convolutions to be substituted.
                        
    Returns:
        A model with Halid operators.
    '''    
    if not C.contrib.netopt.native_convolve_function_registered:
        raise Exception("Could not find {0} library. "
            "Please check if HALIDE_PATH is configured properly "
            "and try building {1} again"
            .format('Cntk.BinaryConvolution-' + C.__version__.rstrip('+'),
            'Extnsibiliy\\BinaryConvolution'))

    bin_conv_filter = (lambda m: type(m) == C.Function 
                and m.is_block 
                and m.op_name == 'BinaryConvolution')        

    def bin_converter(x):

        att = x.block_root.attributes
        # Need a square filter.
        assert(x.inputs[0].shape[-1] == x.inputs[0].shape[-2])

        # These are the attributes needed for the native convolution layer
        # names are defined in BinaryConvolveOp.h
        attributes = {'stride' : att["strides"][-1],
                      'padding' : att["autoPadding"][-1],
                      'size' : x.inputs[0].shape[-1],                       
                      'h' : x.inputs[1].shape[1],
                      'w' : x.inputs[1].shape[2],
                      'channels' :  x.inputs[1].shape[0],
                      'filters' : x.inputs[0].shape[0] }                     
            
        return C.ops.native_user_function(
                    'NativeBinaryConvolveFunction',
                    list(x.inputs), 
                    attributes, 
                    'NativeBinaryConvolution')

    return C.misc.convert(model, bin_conv_filter, bin_converter)


def convert_to_binary_convolution(model, filter_function = None):
    '''
    Replace convolution functions in the model with binary convolutions.
    The function replaces the convolution function inside the 
    cntk.layers.convolution block without changing the block structure.
    The output model has python version of the binarized convolutions.

    Args:
        model           : model that needs to be binarized.
        filter_function : filter layers in the model to apply the binarization
                        
    Returns:
        a model with convolution functions replaced by binary convolutions.
    '''

    convo_block_filter = (lambda x: type(x) == C.Function 
                and x.is_block
                and x.op_name == 'Convolution'
                and (filter_function(x) if filter_function else True))
    
    def convo_block_converter(block):
        
        convo_filter = (lambda x: type(x) == C.Function 
                and not x.is_block # replace the inner function only.
                and x.op_name == 'Convolution')               

        def convolution_converter(x): 

            assert(not x.is_block) # we replace only the function.
            attributes = x.attributes
            # the parameter W of the convolution has the shape 
            # [num filters, depth, (filter shape)]
            num_filters = x.W.shape[0]
            depth = x.W.shape[1]
            filter_shape = (x.W.shape[-2], x.W.shape[-1])
              
            strides = attributes["strides"][-1]
            # check for squre strides for now.
            assert(strides == attributes["strides"][-2]) 
        
            padding =attributes["autoPadding"]
            pad = padding[-1]        
            # Checking for the last two elements in the padding vector. 
            assert(pad == padding[-2])
          
            return  binary_convolution(
                    filter_shape, 
                    num_filters = num_filters, 
                    channels = depth, 
                    strides=strides, 
                    pad = pad,          
                    name='BinaryConvolution')(block.inputs[-1])

        return C.misc.convert(C.as_composite(block.block_root), 
                              convo_filter, 
                              convolution_converter)
    
    return C.misc.convert(model,convo_block_filter,  convo_block_converter)

 
def binary_convolution(filter_shape,
                      num_filters=1,
                      channels = 1,
                      init=C.glorot_uniform(),
                      pad=False,
                      strides=1,                      
                      name='BinaryConvolution'):  
    '''
    Creates a binary convolution function based on the input parameters. 

    Args:
        filter_shape : shape of the filter
        num_filters  : number of filters to use
        init         : initialization function for the filter
        pad          : padding enabled or not for the filter
        strides      : overlap for this filter
        name         : name given to the binary convolution.
                        
    Returns:
        a function for performing binary convolution
    '''

    kernel_shape = (num_filters, channels) + filter_shape
    W = C.Parameter(shape=kernel_shape, init=init, name="filter")
    
    
    def convolution(operand):
        
        bcv_operand_p = C.placeholder(
            operand.shape, operand.dynamic_axes, name="operand")
        
        bcv = C.convolution(
                    CustomMultibit(W, 1), 
                    CustomMultibit(bcv_operand_p, 1), 
                    auto_padding=[False, pad, pad], 
                    strides=[strides])

        return  C.as_block(bcv, [(bcv_operand_p, operand)], name)
                  
    return convolution
