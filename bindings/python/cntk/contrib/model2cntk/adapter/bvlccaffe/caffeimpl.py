# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys

from cntk.contrib.model2cntk.unimodel.cntkmodel import CntkLayerType

CAFFE_LAYER_WRAPPER = {
    'Convolution': CntkLayerType.convolution,
    'BatchNorm': CntkLayerType.batch_normalization,
    'ReLU': CntkLayerType.relu,
    'Pooling': CntkLayerType.pooling,
    'Eltwise_SUM': CntkLayerType.plus,
    'InnerProduct': CntkLayerType.dense,
    'Softmax': CntkLayerType.softmax,
    'Concat': CntkLayerType.splice,
    'SoftmaxWithLoss': CntkLayerType.cross_entropy_with_softmax,
    'Accuracy': CntkLayerType.classification_error,
    'Dropout': CntkLayerType.dropout,
    'LRN': CntkLayerType.lrn,
    'PSROIPooling': CntkLayerType.psroi_pooling
}


def singleton(class_):
    instances = {}

    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return get_instance


@singleton
class CaffeResolver(object):
    def __init__(self):
        self.caffe = None
        self.caffepb = None
        self.net = None
        self.solver = None
        self.__runtime_support__()

    def __runtime_support__(self):
        try:
            import caffe
            self.caffe = caffe
            self.caffepb = caffe.proto.caffe_pb2
        except ImportError:
            sys.stdout.write('using protobuf caffe_pb to load network')
            from adapter.bvlccaffe import caffe_pb2
            self.caffepb = caffe_pb2
        self.net = self.caffepb.NetParameter
        self.solver = self.caffepb.SolverParameter

    def runtime(self):
        return True if self.caffe else False




