import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from cntk import learning_rates_per_sample, DeviceDescriptor, Trainer, sgdlearner
from cntk.ops import variable, constant, parameter, cross_entropy_with_softmax, combine, classification_error, plus, times, relu, convolution, batch_normalization, pooling,AVG_POOLING
from cntk.utils import create_minibatch_source, get_train_loss, cntk_device, create_NDArrayView, create_NDArrayView_from_NumPy
from cntk.examples.common.nn import conv_bn_relu_layer, conv_bn_layer, resnet_node2, resnet_node2_inc

def create_mb_source(epoch_size):    
    image_height = 32
    image_width = 32
    num_channels = 3
    num_classes = 10

    map_file_rel_path = r"../../../../../Examples/Image/Miscellaneous/CIFAR-10/cifar-10-batches-py/train_map.txt"
    map_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), map_file_rel_path)
    mean_file_rel_path = r"../../../../../Examples/Image/Miscellaneous/CIFAR-10/cifar-10-batches-py/CIFAR-10_mean.xml"    
    mean_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), mean_file_rel_path)

    crop_transform_config = dict()
    crop_transform_config["type"] = "Crop"
    crop_transform_config["cropType"] = "Random"
    crop_transform_config["cropRatio"] = "0.8"
    crop_transform_config["jitterType"] = "uniRatio"

    scale_transform_config = dict()
    scale_transform_config["type"] = "Scale"
    scale_transform_config["width"] = image_width
    scale_transform_config["height"] = image_height
    scale_transform_config["channels"] = num_channels
    scale_transform_config["interpolations"] = "linear"

    mean_transform_config = dict()
    mean_transform_config["type"] = "Mean"
    mean_transform_config["meanFile"] = mean_file

    all_transforms = [ crop_transform_config, scale_transform_config, mean_transform_config ]

    features_stream_config = dict()
    features_stream_config["transforms"] = all_transforms

    labels_stream_config = dict()
    labels_stream_config["labelDim"] = num_classes

    input_streams_config = dict()
    input_streams_config["features"] = features_stream_config
    input_streams_config["labels"] = labels_stream_config

    deserializer_config = dict()
    deserializer_config["type"] = "ImageDeserializer"
    deserializer_config["module"] = "ImageReader"
    deserializer_config["file"] = map_file
    deserializer_config["input"] = input_streams_config

    minibatch_config = dict()
    minibatch_config["epochSize"] = epoch_size
    minibatch_config["deserializers"] = [deserializer_config]

    return create_minibatch_source(minibatch_config)    

def get_projection_map(out_dim, in_dim, device):
    if in_dim > out_dim:
        raise ValueError("Can only project from lower to higher dimensionality")
    
    projection_map_values = np.zeros(in_dim*out_dim, dtype=np.float32)
    for i in range(0, in_dim):
        projection_map_values[(i*out_dim) + i] = 1.0
        shape = (in_dim, 1, 1, out_dim)
        return constant(value=projection_map_values.reshape(shape), device_id = device)

def resnet_classifer(input, num_classes, device, output_name):    
    conv_w_scale = 7.07
    conv_b_value = 0

    fc1_w_scale = 0.4
    fc1_b_value = 0

    sc_value = 1
    bn_time_const = 4096

    kernel_width = 3
    kernel_height = 3

    conv1_w_scale = 0.26
    c_map1 = 16    
    
    conv1 = conv_bn_relu_layer(input, c_map1, kernel_width, kernel_height, 1, 1, conv1_w_scale, conv_b_value, sc_value, bn_time_const, device)
    rn1_1 = resnet_node2(conv1, c_map1, kernel_width, kernel_height, conv1_w_scale, conv_b_value, sc_value, bn_time_const, device)
    rn1_2 = resnet_node2(rn1_1, c_map1, kernel_width, kernel_height, conv1_w_scale, conv_b_value, sc_value, bn_time_const, device)
    rn1_3 = resnet_node2(rn1_2, c_map1, kernel_width, kernel_height, conv1_w_scale, conv_b_value, sc_value, bn_time_const, device)
        
    c_map2 = 32
    rn2_1_wProj=get_projection_map(c_map2, c_map1, device)    
    rn2_1 = resnet_node2_inc(rn1_3, c_map2, kernel_width, kernel_height, conv1_w_scale, conv_b_value, sc_value, bn_time_const, rn2_1_wProj, device)
    rn2_2 = resnet_node2(rn2_1, c_map2, kernel_width, kernel_height, conv1_w_scale, conv_b_value, sc_value, bn_time_const, device)
    rn2_3 = resnet_node2(rn2_2, c_map2, kernel_width, kernel_height, conv1_w_scale, conv_b_value, sc_value, bn_time_const, device)
    
    c_map3 = 64
    rn3_1_wProj=get_projection_map(c_map3, c_map2, device)    
    rn3_1 = resnet_node2_inc(rn2_3, c_map3, kernel_width, kernel_height, conv1_w_scale, conv_b_value, sc_value, bn_time_const, rn3_1_wProj, device)
    rn3_2 = resnet_node2(rn3_1, c_map3, kernel_width, kernel_height, conv1_w_scale, conv_b_value, sc_value, bn_time_const, device)
    rn3_3 = resnet_node2(rn3_2, c_map3, kernel_width, kernel_height, conv1_w_scale, conv_b_value, sc_value, bn_time_const, device)

    # Global average pooling    
    poolw = 8
    poolh = 8
    poolh_stride = 1
    poolv_stride = 1

    pool = pooling(rn3_3, AVG_POOLING, (1, poolh, poolw), (1, poolv_stride, poolh_stride))
    out_times_params = parameter(shape=(c_map3, 1, 1, num_classes), device_id=device)
    out_bias_params = parameter(shape=(num_classes), device_id=device)
    t = times(pool, out_times_params)
    return plus(t, out_bias_params, output_name)    

def _test_cifar_resnet():
    dev = 0
    cntk_dev = cntk_device(dev)
    epoch_size = sys.maxsize    
    mbs = create_mb_source(epoch_size)    
    stream_infos = mbs.stream_infos()      
    for si in stream_infos:
        if si.m_name == 'features':
            features_si = si
        elif si.m_name == 'labels':
            labels_si = si

    image_shape = features_si.m_sample_layout.dimensions()          
    image_shape = (image_shape[2], image_shape[0], image_shape[1])
    
    num_classes = labels_si.m_sample_layout.dimensions()[0]
    
    image_input = variable(image_shape, features_si.m_element_type, needs_gradient=False, name="Images")    
    classifer_output = resnet_classifer(image_input, num_classes, dev, "classifierOutput")
    label_var = variable((num_classes), features_si.m_element_type, needs_gradient=False, name="Labels")
    
    ce = cross_entropy_with_softmax(classifer_output, label_var)
    pe = classification_error(classifer_output, label_var)
    image_classifier = combine([ce.owner, pe.owner, classifer_output.owner], "ImageClassifier")

    learning_rate_per_sample = learning_rates_per_sample(0.0078125)
    trainer = Trainer(image_classifier, ce, [sgdlearner(image_classifier.parameters(), learning_rate_per_sample)])
    
    mb_size = 32
    num_mbs = 1000

    minibatch_size_limits = dict()    
    minibatch_size_limits[features_si] = (0,mb_size)
    minibatch_size_limits[labels_si] = (0, mb_size)
    for i in range(0,num_mbs):    
        mb=mbs.get_next_minibatch(minibatch_size_limits, cntk_dev)
        
        arguments = dict()
        arguments[image_input] = mb[features_si].m_data
        arguments[label_var] = mb[labels_si].m_data
        
        trainer.train_minibatch(arguments, cntk_dev)

        freq = 20
        if i % freq == 0:
            print(str(i+freq) + ": " + str(get_train_loss(trainer)))
   
if __name__=='__main__':         
    _test_cifar_resnet()
