
# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
import pytest

from cntk.io import *
from cntk.io import _ReaderConfig
import cntk.io.transforms as xforms

abs_path = os.path.dirname(os.path.abspath(__file__))

AA = np.asarray

def test_text_format(tmpdir):
    mbdata = r'''0	|x 560:1	|y 1 0 0 0 0
0	|x 0:1
0	|x 0:1
1	|x 560:1	|y 0 1 0 0 0
1	|x 0:1
1	|x 0:1
1	|x 424:1
'''
    tmpfile = str(tmpdir/'mbdata.txt')
    with open(tmpfile, 'w') as f:
        f.write(mbdata)

    input_dim = 1000
    num_output_classes = 5

    mb_source = MinibatchSource(CTFDeserializer(tmpfile, StreamDefs(
         features  = StreamDef(field='x', shape=input_dim, is_sparse=True),
         labels    = StreamDef(field='y', shape=num_output_classes, is_sparse=False)
       )))

    assert isinstance(mb_source, MinibatchSource)

    features_si = mb_source.stream_info('features')
    labels_si = mb_source.stream_info('labels')

    mb = mb_source.next_minibatch(7)

    features = mb[features_si]
    # 2 samples, max seq len 4, 1000 dim
    assert features.shape == (2, 4, input_dim)
    assert features.end_of_sweep
    assert features.num_sequences == 2
    assert features.num_samples == 7
    assert features.is_sparse
    # TODO features is sparse and cannot be accessed right now:
    # *** RuntimeError: DataBuffer/WritableDataBuffer methods can only be called for NDArrayiew objects with dense storage format
    # 2 samples, max seq len 4, 1000 dim
    #assert features.data().shape().dimensions() == (2, 4, input_dim)
    #assert features.data().is_sparse()

    labels = mb[labels_si]
    # 2 samples, max seq len 1, 5 dim
    assert labels.shape == (2, 1, num_output_classes)
    assert labels.end_of_sweep
    assert labels.num_sequences == 2
    assert labels.num_samples == 2
    assert not labels.is_sparse

    label_data = np.asarray(labels)
    assert np.allclose(label_data,
            np.asarray([
                [[ 1.,  0.,  0.,  0.,  0.]],
                [[ 0.,  1.,  0.,  0.,  0.]]
                ]))

    mb = mb_source.next_minibatch(1)
    features = mb[features_si]
    labels = mb[labels_si]

    assert not features.end_of_sweep
    assert not labels.end_of_sweep
    assert features.num_samples < 7
    assert labels.num_samples == 1

def test_image():
    map_file = "input.txt"
    mean_file = "mean.txt"
    epoch_size = 150

    feature_name = "f"
    image_width = 100
    image_height = 200
    num_channels = 3

    label_name = "l"
    num_classes = 7

    transforms = [xforms.crop(crop_type='randomside', side_ratio=0.5, jitter_type='uniratio'),
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        xforms.mean(mean_file)]
    image = ImageDeserializer(map_file, StreamDefs(f = StreamDef(field='image', transforms=transforms), l = StreamDef(field='label', shape=num_classes)))

    rc = _ReaderConfig(image, randomize=False, epoch_size=epoch_size)

    assert rc['epochSize'].value == epoch_size
    assert rc['randomize'] == False
    assert rc['sampleBasedRandomizationWindow'] == False
    assert len(rc['deserializers']) == 1
    d = rc['deserializers'][0]
    assert d['type'] == 'ImageDeserializer'
    assert d['file'] == map_file
    assert set(d['input'].keys()) == {label_name, feature_name}

    l = d['input'][label_name]
    assert l['labelDim'] == num_classes

    f = d['input'][feature_name]
    assert set(f.keys()) == { 'transforms' }
    t0, t1, t2 = f['transforms']
    assert t0['type'] == 'Crop'
    assert t1['type'] == 'Scale'
    assert t2['type'] == 'Mean'
    assert t0['cropType'] == 'randomside'
    assert t0['sideRatio'] == 0.5
    assert t0['aspectRatio'] == 1.0
    assert t0['jitterType'] == 'uniratio'
    assert t1['width'] == image_width
    assert t1['height'] == image_height
    assert t1['channels'] == num_channels
    assert t1['interpolations'] == 'linear'
    assert t2['meanFile'] == mean_file

    rc = _ReaderConfig(image, randomize=False, randomization_window = 100,
        sample_based_randomization_window = True, epoch_size=epoch_size)

    assert rc['epochSize'].value == epoch_size
    assert rc['randomize'] == False
    assert rc['sampleBasedRandomizationWindow'] == True
    assert len(rc['deserializers']) == 1
    d = rc['deserializers'][0]
    assert d['type'] == 'ImageDeserializer'
    assert d['file'] == map_file
    assert set(d['input'].keys()) == {label_name, feature_name}

    l = d['input'][label_name]
    assert l['labelDim'] == num_classes

    rc = _ReaderConfig(image, randomize=True, randomization_window = 100,
        sample_based_randomization_window = True, epoch_size=epoch_size)

    assert rc['epochSize'].value == epoch_size
    assert rc['randomize'] == True
    assert rc['sampleBasedRandomizationWindow'] == True
    assert len(rc['deserializers']) == 1
    d = rc['deserializers'][0]
    assert d['type'] == 'ImageDeserializer'
    assert d['file'] == map_file
    assert set(d['input'].keys()) == {label_name, feature_name}

    l = d['input'][label_name]
    assert l['labelDim'] == num_classes

    # TODO depends on ImageReader.dll
    '''
    mbs = rc.minibatch_source()
    sis = mbs.stream_infos()
    assert set(sis.keys()) == { feature_name, label_name }
    '''

def test_full_sweep_minibatch(tmpdir):

    mbdata = r'''0	|S0 0   |S1 0
0	|S0 1 	|S1 1
0	|S0 2
0	|S0 3 	|S1 3
1	|S0 4
1	|S0 5 	|S1 1
1	|S0 6	|S1 2
'''

    tmpfile = str(tmpdir/'mbtest.txt')
    with open(tmpfile, 'w') as f:
        f.write(mbdata)

    mb_source = MinibatchSource(CTFDeserializer(tmpfile, StreamDefs(
        features  = StreamDef(field='S0', shape=1),
        labels    = StreamDef(field='S1', shape=1))),
        randomize=False, epoch_size=FULL_DATA_SWEEP)

    features_si = mb_source.stream_info('features')
    labels_si = mb_source.stream_info('labels')

    mb = mb_source.next_minibatch(1000)
    assert mb[features_si].num_sequences == 2
    assert mb[labels_si].num_sequences == 2

    features = mb[features_si]
    assert features.end_of_sweep
    assert len(features.value) == 2
    expected_features = \
            [
                [[0],[1],[2],[3]],
                [[4],[5],[6]]
            ]

    for res, exp in zip (features.value, expected_features):
        assert np.allclose(res, exp)

    assert np.allclose(features.mask,
            [[2, 1, 1, 1],
             [2, 1, 1, 0]])

    labels = mb[labels_si]
    assert labels.end_of_sweep
    assert len(labels.value) == 2
    expected_labels = \
            [
                [[0],[1],[3]],
                [[1],[2]]
            ]
    for res, exp in zip (labels.value, expected_labels):
        assert np.allclose(res, exp)

    assert np.allclose(labels.mask,
            [[2, 1, 1],
             [2, 1, 0]])

def test_large_minibatch(tmpdir):

    mbdata = r'''0  |S0 0   |S1 0
0   |S0 1   |S1 1
0   |S0 2
0   |S0 3   |S1 3
0   |S0 4
0   |S0 5   |S1 1
0   |S0 6   |S1 2
'''

    tmpfile = str(tmpdir/'mbtest.txt')
    with open(tmpfile, 'w') as f:
        f.write(mbdata)

    mb_source = MinibatchSource(CTFDeserializer(tmpfile, StreamDefs(
        features  = StreamDef(field='S0', shape=1),
        labels    = StreamDef(field='S1', shape=1))),
        randomize=False)

    features_si = mb_source.stream_info('features')
    labels_si = mb_source.stream_info('labels')

    mb = mb_source.next_minibatch(1000)
    features = mb[features_si]
    labels = mb[labels_si]

    # Actually, the minibatch spans over multiple sweeps,
    # not sure if this is an artificial situation, but
    # maybe instead of a boolean flag we should indicate
    # the largest sweep index the data was taken from.
    assert features.end_of_sweep
    assert labels.end_of_sweep

    assert features.num_samples == 1000 - 1000 % 7
    assert labels.num_samples == 5 * (1000 // 7)

    assert mb[features_si].num_sequences == (1000 // 7)
    assert mb[labels_si].num_sequences == (1000 // 7)


@pytest.mark.parametrize("idx, alias_tensor_map, expected", [
    (0, {'A': [object()]}, ValueError),
])
def test_sequence_conversion_exceptions(idx, alias_tensor_map, expected):
    with pytest.raises(expected):
        sequence_to_cntk_text_format(idx, alias_tensor_map)


@pytest.mark.parametrize("idx, alias_tensor_map, expected", [
    (0, {'W': AA([])}, ""),
    (0, {'W': AA([[[1, 0, 0, 0], [1, 0, 0, 0]]])}, """\
0\t|W 1 0 0 0 1 0 0 0\
"""),
    (0, {
        'W': AA([[[1, 0, 0, 0], [1, 0, 0, 0]]]),
        'L': AA([[[2]]])
    },
        """\
0\t|L 2 |W 1 0 0 0 1 0 0 0\
"""),
    (0, {
        'W': AA([[[1, 0], [1, 0]], [[5, 6], [7, 8]]]),
        'L': AA([[[2]]])
    },
        """\
0\t|L 2 |W 1 0 1 0
0\t|W 5 6 7 8"""),
])
def test_sequence_conversion_dense(idx, alias_tensor_map, expected):
    assert sequence_to_cntk_text_format(idx, alias_tensor_map) == expected


@pytest.mark.parametrize("data, expected", [
    ([1], True),
    ([[1, 2]], True),
    ([[AA([1, 2])]], False),
    ([AA([1, 2])], False),
    ([AA([1, 2]), AA([])], False),
])
def test_is_tensor(data, expected):
    from cntk.io import _is_tensor
    assert _is_tensor(data) == expected


def test_create_two_image_deserializers(tmpdir):
    mbdata = r'''filename	0
filename2	0
'''

    map_file = str(tmpdir/'mbdata.txt')
    with open(map_file, 'w') as f:
        f.write(mbdata)

    image_width = 100
    image_height = 200
    num_channels = 3
    num_classes = 7

    transforms = [xforms.crop(crop_type='randomside', side_ratio=0.5, jitter_type='uniratio'),
                  xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')]
        
    image1 = ImageDeserializer(map_file, StreamDefs(f1 = StreamDef(field='image', transforms=transforms)))
    image2 = ImageDeserializer(map_file, StreamDefs(f2 = StreamDef(field='image', transforms=transforms)))

    mb_source = MinibatchSource([image1, image2])
    assert isinstance(mb_source, MinibatchSource)
