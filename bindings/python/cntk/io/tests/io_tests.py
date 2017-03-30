
# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
import pytest

from cntk.io import *
import cntk.io.transforms as xforms
from cntk.cntk_py import to_dictionary
from cntk.cntk_py import MinibatchSourceConfig

abs_path = os.path.dirname(os.path.abspath(__file__))

AA = np.asarray

def create_temp_file(tmpdir):
    tmpfile = str(tmpdir/'mbtest.txt')
    with open(tmpfile, 'w') as f:
        f.write("|S0 1\n|S0 2\n|S0 3\n|S0 4")
    return tmpfile

def create_ctf_deserializer(tmpdir):
    tmpfile = create_temp_file(tmpdir)
    return CTFDeserializer(tmpfile, StreamDefs(features  = StreamDef(field='S0', shape=1)))

def create_config(tmpdir):
    tmpfile = create_temp_file(tmpdir)
    return MinibatchSourceConfig() \
        .add_deserializer(
            CTFDeserializer(tmpfile, 
                StreamDefs(features  = StreamDef(field='S0', shape=1))))


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

    labels = mb[labels_si]
    # 2 samples, max seq len 1, 5 dim
    assert labels.shape == (2, 1, num_output_classes)
    assert labels.end_of_sweep
    assert labels.num_sequences == 2
    assert labels.num_samples == 2
    assert not labels.is_sparse

    label_data = labels.asarray()
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

def check_default_config_keys(d):
        assert 5 <= len(d.keys())
        assert False == d['frameMode']
        assert False == d['multiThreadedDeserialization']
        assert TraceLevel.Warning == d['traceLevel']
        assert 'randomize' in d.keys()
        assert 'deserializers' in d.keys()

def test_minibatch_source_config_constructor(tmpdir):
    ctf = create_ctf_deserializer(tmpdir)

    config = MinibatchSourceConfig([ctf], False)
    dictionary = to_dictionary(config)
    check_default_config_keys(dictionary)
    assert 5 == len(dictionary.keys())
    assert False == dictionary['randomize']

    config = MinibatchSourceConfig([ctf], True)
    dictionary = to_dictionary(config)
    check_default_config_keys(dictionary)

    assert 7 == len(dictionary.keys())
    assert True == dictionary['randomize']
    assert DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS == dictionary['randomizationWindow']
    assert False == dictionary['sampleBasedRandomizationWindow']

    config = MinibatchSourceConfig([ctf]) # 'randomize' is omitted
    dictionary = to_dictionary(config)
    check_default_config_keys(dictionary)

    assert 7 == len(dictionary.keys())
    assert True == dictionary['randomize']
    assert DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS == dictionary['randomizationWindow']
    assert False == dictionary['sampleBasedRandomizationWindow']

def test_minibatch_source_config_sweeps_and_samples(tmpdir):
    ctf = create_ctf_deserializer(tmpdir)
    config = MinibatchSourceConfig([ctf])

    assert INFINITELY_REPEAT == config.max_samples
    assert INFINITELY_REPEAT == config.max_sweeps

    config.max_samples = 100
    config.max_sweeps = 3
    assert 100 == config.max_samples
    assert 3 == config.max_sweeps
    
    with pytest.raises(Exception):
        # to_dictionary will validate the config
        dictionary = to_dictionary(config)

    config.max_samples = INFINITELY_REPEAT
    dictionary = to_dictionary(config)
    check_default_config_keys(dictionary)

def test_minibatch_source_config_randomization(tmpdir):
    ctf = create_ctf_deserializer(tmpdir)
    config = MinibatchSourceConfig([ctf])

    dictionary = to_dictionary(config)
    check_default_config_keys(dictionary)
    assert True == dictionary['randomize']

    config.randomization_window_in_chunks = 0
    dictionary = to_dictionary(config)
    check_default_config_keys(dictionary)
    assert False == dictionary['randomize']

    config.randomization_window_in_chunks = 10
    dictionary = to_dictionary(config)
    check_default_config_keys(dictionary)
    assert True == dictionary['randomize']
    assert 10 == dictionary['randomizationWindow']
    assert False == dictionary['sampleBasedRandomizationWindow']

    config.randomization_window_in_samples = 100
    with pytest.raises(Exception):
        # to_dictionary will validate the config
        dictionary = to_dictionary(config)

    config.randomization_window_in_chunks = 0
    dictionary = to_dictionary(config)
    check_default_config_keys(dictionary)
    assert True == dictionary['randomize']
    assert 100 == dictionary['randomizationWindow']
    assert True == dictionary['sampleBasedRandomizationWindow']
    
def test_minibatch_source_config_other_properties(tmpdir):
    ctf = create_ctf_deserializer(tmpdir)
    config = MinibatchSourceConfig([ctf])

    config.is_multithreaded = True
    config.trace_level = TraceLevel.Info.value
    config.is_frame_mode_enabled = True

    dictionary = to_dictionary(config)
    assert 7 == len(dictionary.keys())
    assert TraceLevel.Info == dictionary['traceLevel']
    assert True == dictionary['frameMode']
    assert True == dictionary['multiThreadedDeserialization']

    config.is_multithreaded = False
    config.trace_level = 0
    config.truncation_length = 123
    with pytest.raises(Exception):
        # to_dictionary will validate the config
        dictionary = to_dictionary(config)

    config.is_frame_mode_enabled = False

    dictionary = to_dictionary(config)
    assert 9 == len(dictionary.keys())
    assert 0 == dictionary['traceLevel']
    assert False == dictionary['frameMode']
    assert False == dictionary['multiThreadedDeserialization']
    assert True == dictionary['truncated']
    assert 123 == dictionary['truncationLength']

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

    config = to_dictionary(MinibatchSourceConfig([image], randomize=False))
    
    assert len(config['deserializers']) == 1
    d = config['deserializers'][0]
    assert d['type'] == 'ImageDeserializer'
    assert d['file'] == map_file
    assert set(d['input'].keys()) == {label_name, feature_name}

    l = d['input'][label_name]
    assert l['labelDim'] == num_classes

    f = d['input'][feature_name]
    assert set(f.keys()) == { 'transforms' }
    t0, t1, t2, _ = f['transforms']
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

    
    config = to_dictionary(MinibatchSourceConfig([image, image]))
    assert len(config['deserializers']) == 2

    config = to_dictionary(MinibatchSourceConfig([image, image, image]))
    assert len(config['deserializers']) == 3

    # TODO depends on ImageReader.dll
    '''
    mbs = config.create_minibatch_source()
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
        randomization_window_in_chunks=0, max_sweeps=1)

    features_si = mb_source.stream_info('features')
    labels_si = mb_source.stream_info('labels')

    mb = mb_source.next_minibatch(1000)

    assert mb[features_si].num_sequences == 2
    assert mb[labels_si].num_sequences == 2

    features = mb[features_si]
    assert features.end_of_sweep
    assert len(features.as_sequences()) == 2
    expected_features = \
            [
                [[0],[1],[2],[3]],
                [[4],[5],[6]]
            ]

    for res, exp in zip (features.as_sequences(), expected_features):
        assert np.allclose(res, exp)

    assert np.allclose(features.data.mask,
            [[2, 1, 1, 1],
             [2, 1, 1, 0]])

    labels = mb[labels_si]
    assert labels.end_of_sweep
    assert len(labels.as_sequences()) == 2
    expected_labels = \
            [
                [[0],[1],[3]],
                [[1],[2]]
            ]
    for res, exp in zip (labels.as_sequences(), expected_labels):
        assert np.allclose(res, exp)

    assert np.allclose(labels.data.mask,
            [[2, 1, 1],
             [2, 1, 0]])

def test_max_samples(tmpdir):
    mb_source = MinibatchSource(
        create_ctf_deserializer(tmpdir), max_samples=1)

    input_map = {'features' : mb_source['features']}
    mb = mb_source.next_minibatch(10, input_map)

    assert 'features' in mb
    assert mb['features'].num_samples == 1
    assert not mb['features'].end_of_sweep

    mb = mb_source.next_minibatch(10, input_map)

    assert not mb

def test_max_sweeps(tmpdir):
    # set max sweeps to 3 (12 samples altogether).
    mb_source = MinibatchSource(
        create_ctf_deserializer(tmpdir), max_sweeps=3)

    input_map = {'features' : mb_source['features']}

    for i in range(2):
        mb = mb_source.next_minibatch(5, input_map)

        assert 'features' in mb
        assert mb['features'].num_samples == 5
        assert mb['features'].end_of_sweep

    mb = mb_source.next_minibatch(5, input_map)

    assert 'features' in mb
    assert mb['features'].num_samples == 2
    assert mb['features'].end_of_sweep

    mb = mb_source.next_minibatch(1, input_map)

    assert not mb

def test_max_samples_over_several_sweeps(tmpdir):
    mb_source = MinibatchSource(
        create_ctf_deserializer(tmpdir), max_samples=11)

    input_map = {'features' : mb_source['features']}

    for i in range(2):
        mb = mb_source.next_minibatch(5, input_map)

        assert 'features' in mb
        assert mb['features'].num_samples == 5
        assert mb['features'].end_of_sweep

    mb = mb_source.next_minibatch(5, input_map)

    assert 'features' in mb
    assert mb['features'].num_samples == 1
    assert not mb['features'].end_of_sweep

    mb = mb_source.next_minibatch(1, input_map)

    assert not mb

def test_one_sweep(tmpdir):
    ctf = create_ctf_deserializer(tmpdir)
    sources = [ MinibatchSource(ctf, max_sweeps=1),
                MinibatchSource(ctf, max_samples=FULL_DATA_SWEEP),
                MinibatchSource(ctf, max_sweeps=1,
                    max_samples=INFINITELY_REPEAT),
                MinibatchSource(ctf, max_samples=FULL_DATA_SWEEP,
                    max_sweeps=INFINITELY_REPEAT) ]

    for source in sources:
        input_map = {'features' : source['features']}

        mb = source.next_minibatch(100, input_map)

        assert 'features' in mb
        assert mb['features'].num_samples == 4
        assert mb['features'].end_of_sweep

        mb = source.next_minibatch(100, input_map)

        assert not mb

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
        randomization_window_in_chunks=0)

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
