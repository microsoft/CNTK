
# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import cntk as C
import pytest

from cntk.io import MinibatchSource, CTFDeserializer, CBFDeserializer, \
    StreamDefs, StreamDef, \
    ImageDeserializer, Base64ImageDeserializer, \
    FULL_DATA_SWEEP, INFINITELY_REPEAT, \
    DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS, \
    sequence_to_cntk_text_format, UserMinibatchSource, StreamInformation, \
    MinibatchData, UserDeserializer
from cntk.ops.tests.ops_test_utils import cntk_device
from cntk.logging import TraceLevel
import cntk.io.transforms as xforms
from cntk.cntk_py import to_dictionary, MinibatchSourceConfig
from cntk.core import Value

AA = np.asarray

MBDATA_DENSE_1 = r'''0  |S0 0   |S1 0
0   |S0 1   |S1 1
0   |S0 2
0   |S0 3   |S1 3
1   |S0 4
1   |S0 5   |S1 1
1   |S0 6   |S1 2
'''

MBDATA_DENSE_2 = r'''0  |S0 0   |S1 0
0   |S0 1   |S1 1
0   |S0 2
0   |S0 3   |S1 3
0   |S0 4
0   |S0 5   |S1 1
0   |S0 6   |S1 2
'''

MBDATA_SPARSE = r'''0	|x 560:1	|y 1 0 0 0 0
0	|x 0:1
0	|x 0:1
1	|x 560:1	|y 0 1 0 0 0
1	|x 0:1
1	|x 0:1
1	|x 424:1
'''

MBDATA_SPARSE1 = r'''0	|x 560:1
0	|x 0:1
0	|x 0:1
1	|x 560:1
1	|x 0:1
1	|x 0:1
1	|x 424:1
'''

MBDATA_SPARSE2 = r'''0	|y 1 0 0 0 0
1	|y 0 1 0 0 0
'''

def create_temp_file(tmpdir):
    tmpfile = str(tmpdir/'mbtest.txt')
    with open(tmpfile, 'w') as f:
        f.write("|S0 1\n|S0 2\n|S0 3\n|S0 4")
    return tmpfile


def create_ctf_deserializer(tmpdir):
    tmpfile = create_temp_file(tmpdir)
    return CTFDeserializer(tmpfile, StreamDefs(features=StreamDef(field='S0', shape=1)))


def create_config(tmpdir):
    tmpfile = create_temp_file(tmpdir)
    return MinibatchSourceConfig() \
        .add_deserializer(
            CTFDeserializer(tmpfile,
                StreamDefs(features=StreamDef(field='S0', shape=1))))


def _write_data(tmpdir, data, filename='mbdata.txt'):
    tmpfile = str(tmpdir / filename)

    with open(tmpfile, 'w') as f:
        f.write(data)

    return tmpfile


def test_text_format(tmpdir):
    tmpfile = _write_data(tmpdir, MBDATA_SPARSE)

    input_dim = 1000
    num_output_classes = 5

    mb_source = MinibatchSource(CTFDeserializer(tmpfile, StreamDefs(
        features=StreamDef(field='x', shape=input_dim, is_sparse=True),
        labels=StreamDef(field='y', shape=num_output_classes, is_sparse=False)
    )), randomize=False)

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
                           [[1.,  0.,  0.,  0.,  0.]],
                           [[0.,  1.,  0.,  0.,  0.]]
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
        assert d['frameMode'] is False
        assert d['multiThreadedDeserialization'] is False
        assert TraceLevel.Warning == d['traceLevel']
        assert 'randomize' in d.keys()
        assert 'deserializers' in d.keys()


def test_minibatch_source_config_constructor(tmpdir):
    ctf = create_ctf_deserializer(tmpdir)

    config = MinibatchSourceConfig([ctf], False)
    dictionary = to_dictionary(config)
    check_default_config_keys(dictionary)
    assert 5 == len(dictionary.keys())
    assert dictionary['randomize'] is False

    config = MinibatchSourceConfig([ctf], True)
    dictionary = to_dictionary(config)
    check_default_config_keys(dictionary)

    assert 8 == len(dictionary.keys())
    assert dictionary['randomize'] is True
    assert DEFAULT_RANDOMIZATION_WINDOW_IN_CHUNKS == dictionary['randomizationWindow']
    assert False == dictionary['sampleBasedRandomizationWindow']

    config = MinibatchSourceConfig([ctf]) # 'randomize' is omitted
    dictionary = to_dictionary(config)
    check_default_config_keys(dictionary)

    assert 8 == len(dictionary.keys())
    assert dictionary['randomize'] is True
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
    assert dictionary['randomize'] is True

    config.randomization_window_in_chunks = 0
    dictionary = to_dictionary(config)
    check_default_config_keys(dictionary)
    assert dictionary['randomize'] is False

    config.randomization_window_in_chunks = 10
    dictionary = to_dictionary(config)
    check_default_config_keys(dictionary)
    assert dictionary['randomize'] is True
    assert 10 == dictionary['randomizationWindow']
    assert dictionary['sampleBasedRandomizationWindow'] is False

    config.randomization_window_in_samples = 100
    with pytest.raises(Exception):
        # to_dictionary will validate the config
        dictionary = to_dictionary(config)

    config.randomization_window_in_chunks = 0
    dictionary = to_dictionary(config)
    check_default_config_keys(dictionary)
    assert dictionary['randomize'] is True
    assert 100 == dictionary['randomizationWindow']
    assert dictionary['sampleBasedRandomizationWindow'] is True


def test_minibatch_source_config_other_properties(tmpdir):
    ctf = create_ctf_deserializer(tmpdir)
    config = MinibatchSourceConfig([ctf])

    config.is_multithreaded.set(True)
    config.trace_level = TraceLevel.Info.value
    config.is_frame_mode_enabled = True

    dictionary = to_dictionary(config)
    assert 8 == len(dictionary.keys())
    assert TraceLevel.Info == dictionary['traceLevel']
    assert dictionary['frameMode'] is True
    assert dictionary['multiThreadedDeserialization'] is True

    config.is_multithreaded.set(False)
    config.trace_level = 0
    config.truncation_length = 123
    with pytest.raises(Exception):
        # to_dictionary will validate the config
        dictionary = to_dictionary(config)

    config.is_frame_mode_enabled = False

    dictionary = to_dictionary(config)
    assert 10 == len(dictionary.keys())
    assert 0 == dictionary['traceLevel']
    assert dictionary['frameMode'] is False
    assert dictionary['multiThreadedDeserialization'] is False
    assert dictionary['truncated'] is True
    assert 123 == dictionary['truncationLength']


def test_image(tmpdir):
    map_file = "input.txt"
    mean_file = "mean.txt"

    feature_name = "f"
    image_width = 100
    image_height = 200
    num_channels = 3

    label_name = "l"
    num_classes = 7

    transforms = [
        xforms.crop(crop_type='randomside', side_ratio=0.5,
                    jitter_type='uniratio'),
        xforms.scale(width=image_width, height=image_height,
                     channels=num_channels, interpolations='linear'),
        xforms.mean(mean_file)]
    defs = StreamDefs(f=StreamDef(field='image', transforms=transforms),
                      l=StreamDef(field='label', shape=num_classes))
    image = ImageDeserializer(map_file, defs)

    config = to_dictionary(MinibatchSourceConfig([image], randomize=False))

    # Multithreading should be on by default for the ImageDeserializer.
    assert config['multiThreadedDeserialization'] is True
    assert len(config['deserializers']) == 1

    d = config['deserializers'][0]
    assert d['type'] == 'ImageDeserializer'
    assert d['file'] == map_file
    assert set(d['input'].keys()) == {label_name, feature_name}

    l = d['input'][label_name]
    assert l['labelDim'] == num_classes

    f = d['input'][feature_name]
    assert set(f.keys()) == {'transforms'}
    t0, t1, t2, _ = f['transforms']
    assert t0['type'] == 'Crop'
    assert t1['type'] == 'Scale'
    assert t2['type'] == 'Mean'
    assert t0['cropType'] == 'randomside'
    assert t0['cropSize'] == '0:0'
    assert t0['sideRatio'] == '0.5:0.5'
    assert t0['aspectRatio'] == '1:1'
    assert t0['areaRatio'] == '0:0'
    assert t0['jitterType'] == 'uniratio'
    assert t1['width'] == image_width
    assert t1['height'] == image_height
    assert t1['channels'] == num_channels
    assert t1['interpolations'] == 'linear'
    assert t2['meanFile'] == mean_file

    config = to_dictionary(MinibatchSourceConfig([image, image]))
    assert len(config['deserializers']) == 2

    ctf = create_ctf_deserializer(tmpdir)
    config = to_dictionary(MinibatchSourceConfig([image, ctf, image]))
    # Multithreading should still be enabled.
    assert config['multiThreadedDeserialization'] is True
    assert len(config['deserializers']) == 3



    # TODO depends on ImageReader.dll
    '''
    mbs = config.create_minibatch_source()
    sis = mbs.stream_infos()
    assert set(sis.keys()) == { feature_name, label_name }
    '''

def test_image_with_crop_range():
    map_file = "input.txt"

    feature_name = "f"
    image_width = 100
    image_height = 200
    num_channels = 3

    label_name = "l"
    num_classes = 7

    transforms = [
        xforms.crop(crop_type='randomside', 
                    crop_size=(512,424), side_ratio=(0.2, 0.5), area_ratio=(0.1, 0.75), aspect_ratio=(0.3, 0.8),
                    jitter_type='uniratio')
        ]
    defs = StreamDefs(f=StreamDef(field='image', transforms=transforms),
                      l=StreamDef(field='label', shape=num_classes))
    image = ImageDeserializer(map_file, defs)

    config = to_dictionary(MinibatchSourceConfig([image], randomize=False))

    assert len(config['deserializers']) == 1
    d = config['deserializers'][0]
    assert d['type'] == 'ImageDeserializer'
    assert d['file'] == map_file
    assert set(d['input'].keys()) == {label_name, feature_name}

    l = d['input'][label_name]
    assert l['labelDim'] == num_classes

    f = d['input'][feature_name]
    assert set(f.keys()) == {'transforms'}
    t0,  _ = f['transforms']
    assert t0['type'] == 'Crop'
    assert t0['cropType'] == 'randomside'
    assert t0['cropSize'] == '512:424'
    assert t0['sideRatio'] == '0.2:0.5'
    assert t0['aspectRatio'] == '0.3:0.8'
    assert t0['areaRatio'] == '0.1:0.75'
    assert t0['jitterType'] == 'uniratio'

    config = to_dictionary(MinibatchSourceConfig([image, image]))
    assert len(config['deserializers']) == 2

    config = to_dictionary(MinibatchSourceConfig([image, image, image]))
    assert len(config['deserializers']) == 3

def test_full_sweep_minibatch(tmpdir):
    tmpfile = _write_data(tmpdir, MBDATA_DENSE_1)

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
            [[0], [1], [2], [3]],
            [[4], [5], [6]]
        ]

    for res, exp in zip(features.as_sequences(), expected_features):
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
    for res, exp in zip(labels.as_sequences(), expected_labels):
        assert np.allclose(res, exp)

    assert np.allclose(labels.data.mask,
            [[2, 1, 1],
             [2, 1, 0]])


def test_max_samples(tmpdir):
    mb_source = MinibatchSource(
        create_ctf_deserializer(tmpdir), max_samples=1)

    input_map = {'features': mb_source['features']}
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

    input_map = {'features': mb_source['features']}

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

    input_map = {'features': mb_source['features']}

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
    sources = [MinibatchSource(ctf, max_sweeps=1),
               MinibatchSource(ctf, max_samples=FULL_DATA_SWEEP),
               MinibatchSource(ctf, max_sweeps=1, max_samples=INFINITELY_REPEAT),
               MinibatchSource(ctf, max_samples=FULL_DATA_SWEEP, max_sweeps=INFINITELY_REPEAT)]

    for source in sources:
        input_map = {'features': source['features']}

        mb = source.next_minibatch(100, input_map)

        assert 'features' in mb
        assert mb['features'].num_samples == 4
        assert mb['features'].end_of_sweep

        mb = source.next_minibatch(100, input_map)

        assert not mb

def test_random_seed(tmpdir):
    ctf = create_ctf_deserializer(tmpdir)
    sources = [MinibatchSource(ctf),
               MinibatchSource(ctf, randomization_seed=123),
               MinibatchSource(ctf, randomization_seed=0),
               MinibatchSource(ctf, randomization_seed=1)]

    data = []

    for source in sources:
        input_map = {'features': source['features']}

        mb = source.next_minibatch(100, input_map)
        data.append(mb['features'].asarray())

    assert not (data[0] == data[1]).all()
    assert (data[0] == data[2]).all()
    # after the first sweep (= 4 samples), the first reader is seeded
    # with 1, and should produce results identical to the last reader.
    assert (data[0][4:] == data[3][:-4]).all()


def test_large_minibatch(tmpdir):
    tmpfile = _write_data(tmpdir, MBDATA_DENSE_2)

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

    map_file = str(tmpdir / 'mbdata.txt')
    with open(map_file, 'w') as f:
        f.write(mbdata)

    image_width = 100
    image_height = 200
    num_channels = 3

    transforms = [xforms.crop(crop_type='randomside', side_ratio=0.5,
                              jitter_type='uniratio'),
                  xforms.scale(width=image_width, height=image_height,
                               channels=num_channels, interpolations='linear')]

    image1 = ImageDeserializer(
        map_file, StreamDefs(f1=StreamDef(field='image',
                             transforms=transforms)))
    image2 = ImageDeserializer(
        map_file, StreamDefs(f2=StreamDef(field='image',
                             transforms=transforms)))

    mb_source = MinibatchSource([image1, image2])
    assert isinstance(mb_source, MinibatchSource)


def test_base64_image_deserializer(tmpdir):
    import io, base64, uuid; from PIL import Image
    images, b64_images = [], []

    np.random.seed(1)
    for i in range(10):
        data = np.random.randint(0, 2**8, (5,7,3))
        image = Image.fromarray(data.astype('uint8'), "RGB")
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        assert image.width == 7 and image.height == 5
        b64_images.append(base64.b64encode(buf.getvalue()))
        images.append(np.array(image))

    image_data = str(tmpdir / 'mbdata1.txt')
    seq_ids = []
    uid = uuid.uuid1().int >> 64
    with open(image_data, 'wb') as f:
        for i,data in enumerate(b64_images):
            seq_id = uid ^ i
            seq_id = str(seq_id).encode('ascii')
            seq_ids.append(seq_id)
            line = seq_id + b'\t'
            label = str(i).encode('ascii')
            line += label + b'\t' + data + b'\n'
            f.write(line)

    ctf_data = str(tmpdir / 'mbdata2.txt')
    with open(ctf_data, 'wb') as f:
        for i, sid in enumerate(seq_ids):
            line = sid + b'\t' + b'|index '+str(i).encode('ascii') + b'\n'
            f.write(line)

    transforms = [xforms.scale(width=7, height=5, channels=3)]
    b64_deserializer = Base64ImageDeserializer(image_data,
        StreamDefs(
            images=StreamDef(field='image', transforms=transforms),
            labels=StreamDef(field='label', shape=10)))

    ctf_deserializer = CTFDeserializer(ctf_data,
        StreamDefs(index=StreamDef(field='index', shape=1)))

    mb_source = MinibatchSource([ctf_deserializer, b64_deserializer])
    assert isinstance(mb_source, MinibatchSource)

    for j in range(100):
        mb = mb_source.next_minibatch(10)

        index_stream = mb_source.streams['index']
        index = mb[index_stream].asarray().flatten()
        image_stream = mb_source.streams['images']

        results = mb[image_stream].asarray()

        for i in range(10):
            # original images are RBG, openCV produces BGR images,
            # reverse the last dimension of the original images
            bgrImage = images[int(index[i])][:,:,::-1]
            # transposing to get CHW representation
            bgrImage = np.transpose(bgrImage, (2, 0, 1))
            assert (bgrImage == results[i][0]).all()

class MyDataSource(UserMinibatchSource):
    def __init__(self, f_dim, l_dim):
        self.f_dim, self.l_dim = f_dim, l_dim

        self.fsi = StreamInformation("features", 0, 'sparse', np.float32, (self.f_dim,))
        self.lsi = StreamInformation("labels", 1, 'dense', np.float32, (self.l_dim,))

        # MBDATA_SPARSE fits into memory we will, so we will read it in all at
        # once. It follows the CNTKTextFormat:
        #   sequence ID |feature1 data |feature2 data
        # where in this case feature1's data is encoded as one-hot and we will
        # convert to CSR, and feature2's data is a one-hot encoded as dense.

        # We will store
        #   sequence id -> "features" -> list of features
        # and
        #   sequence id -> "labels" -> label

        self.data = {}
        for line in MBDATA_SPARSE.split('\n'):
            line = line.strip()
            if not line:
                continue
            seq_id, data = line.split('|', 1)
            data = data.split("|")
            seq_id = int(seq_id.strip())

            if seq_id not in self.data:
                self.data[seq_id] = {'features': []}

            # Processing features - expecting one per line.
            # We accumulate the vocabulary indices and convert them into a
            # Value object when requested in next_minibatch()
            features = data[0].split(" ")
            assert features[0] == 'x'
            vocab_idx = int(features[1].split(":")[0])
            self.data[seq_id]['features'].append(vocab_idx)

            # Process label, if exists
            if len(data) == 2:
                # Only one label definition per sequence allowed
                assert 'labels' not in self.data[seq_id]

                labels = data[1].split(" ")
                assert labels[0] == 'y'
                # We don't have many label classes, and only one label per
                # sequence, so we just read it in as dense, all at once.
                val = np.asarray([labels[1:]], dtype=np.float32)
                self.data[seq_id]['labels'] = val

        self.sequences = sorted(self.data)
        self.next_seq_idx = 0

        super(MyDataSource, self).__init__()

    def stream_infos(self):
        return [self.fsi, self.lsi]

    def next_minibatch(self, num_samples, number_of_workers, worker_rank, device=None):
        features = []
        labels = []

        sweep_end = False

        f_sample_count = 0
        l_sample_count = 0


        while max(f_sample_count, l_sample_count) < num_samples:
            if self.next_seq_idx == len(self.sequences):
                sweep_end = True
                self.next_seq_idx = 0

            seq_id = self.sequences[self.sequences[self.next_seq_idx]]

            f_data = self.data[seq_id]['features']
            l_data = self.data[seq_id]['labels']
            if (features or labels) and max(f_sample_count+len(f_data), l_sample_count+len(l_data)) > num_samples:
                break
            f_sample_count += len(f_data)
            features.append(f_data)

            l_sample_count += len(l_data)
            labels.append(l_data)

            self.next_seq_idx += 1

        num_seq = len(features)

        f_data = Value.one_hot(batch=features, num_classes=self.f_dim)
        l_data = Value(batch=np.asarray(labels, dtype=np.float32))
        result = {
                self.fsi: MinibatchData(f_data, num_seq, f_sample_count, sweep_end),
                self.lsi: MinibatchData(l_data, num_seq, l_sample_count, sweep_end)
                }

        return result


class MyDataSourceWithCheckpoint(MyDataSource):
    def __init__(self, f_dim, l_dim):
        super(MyDataSourceWithCheckpoint, self).__init__(f_dim, l_dim)
        self._restore_from_checkpoint_calls = 0

    def get_checkpoint_state(self):
        return {'test': 12}

    def restore_from_checkpoint(self, state):
        self._restore_from_checkpoint_calls += 1
        assert state == {'test': 12}


def test_usermbsource(tmpdir):
    tmpfile = _write_data(tmpdir, MBDATA_SPARSE)

    input_dim = 1000
    num_output_classes = 5

    # Setting up the native MB source as the ground truth
    n_mb_source = CTFDeserializer(tmpfile, StreamDefs(
        features=StreamDef(field='x', shape=input_dim, is_sparse=True),
        labels=StreamDef(field='y', shape=num_output_classes, is_sparse=False)
    ))
    n_mb_source = MinibatchSource(n_mb_source, randomize=False)
    n_features_si = n_mb_source['features']
    n_labels_si = n_mb_source['labels']

    n_mb = n_mb_source.next_minibatch(2)
    n_features = n_mb[n_features_si]
    n_labels = n_mb[n_labels_si]

    # Setting up the user MB source
    u_mb_source = MyDataSource(input_dim, num_output_classes)
    u_features_si = u_mb_source['features']
    u_labels_si = u_mb_source['labels']

    u_mb = u_mb_source.next_minibatch(2, 1, 0)
    u_features = u_mb[u_features_si]
    u_labels = u_mb[u_labels_si]

    assert u_features.shape == n_features.shape == (1, 3, 1000)
    assert u_features.end_of_sweep == n_features.end_of_sweep
    assert u_features.num_sequences == n_features.num_sequences
    assert u_features.num_samples == n_features.num_samples
    assert u_features.is_sparse == n_features.is_sparse

    assert u_labels.shape == n_labels.shape == (1, 1, 5)
    assert u_labels.end_of_sweep is n_labels.end_of_sweep is False
    assert u_labels.num_sequences == u_labels.num_sequences
    assert u_labels.num_samples == u_labels.num_samples
    assert u_labels.is_sparse is n_labels.is_sparse is False

    u_label_data = u_labels.asarray()
    n_label_data = n_labels.asarray()
    assert np.allclose(u_label_data, n_label_data)

    n_mb = n_mb_source.next_minibatch(10)
    n_features = n_mb[n_features_si]
    n_labels = n_mb[n_labels_si]

    u_mb = u_mb_source.next_minibatch(10, 1, 0)
    u_features = u_mb[u_features_si]
    u_labels = u_mb[u_labels_si]

    assert u_labels.shape == n_labels.shape
    u_label_data = u_labels.asarray()
    n_label_data = n_labels.asarray()

    assert np.allclose(u_label_data, n_label_data)

    assert u_features.end_of_sweep is u_labels.end_of_sweep is True
    assert u_features.num_samples == n_features.num_samples
    assert u_features.num_sequences == n_features.num_sequences


@pytest.mark.parametrize("with_checkpoint_impl", [True, False])
def test_usermbsource_training(tmpdir, with_checkpoint_impl):
    input_dim = 1000
    num_output_classes = 5

    mbs = MyDataSource(input_dim, num_output_classes)
    # Using this for testing the UserMinibatchSource checkpointing
    if with_checkpoint_impl:
        MBS_CV_CLASS = MyDataSourceWithCheckpoint
    else:
        MBS_CV_CLASS = MyDataSource

    mbs_cv = MBS_CV_CLASS(input_dim, num_output_classes)

    from cntk import sequence, parameter, plus, cross_entropy_with_softmax, \
            classification_error, learning_rate_schedule, sgd, Trainer, \
            training_session, times, UnitType

    feature = sequence.input_variable(shape=(input_dim,))
    label = C.input_variable(shape=(num_output_classes,))
    p = parameter(shape=(input_dim, num_output_classes), init=10)
    z = times(sequence.reduce_sum(feature), p, name='z')
    ce = cross_entropy_with_softmax(z, label)
    errs = classification_error(z, label)

    lr_per_sample = learning_rate_schedule(
        [0.3, 0.2, 0.1, 0.0], UnitType.sample)
    learner = sgd(z.parameters, lr_per_sample)
    trainer = Trainer(z, (ce, errs), [learner])
    input_map = {
        feature: mbs.fsi,
        label: mbs.lsi
    }

    session = training_session(
        trainer=trainer, mb_source=mbs,
        model_inputs_to_streams=input_map,
        mb_size=4, max_samples=20,
        cv_config = C.CrossValidationConfig(minibatch_source=mbs_cv, max_samples=10,
            minibatch_size=2)
    )
    session.train()

    assert trainer.total_number_of_samples_seen == 20
    if with_checkpoint_impl:
        assert mbs_cv._restore_from_checkpoint_calls == 1


def test_minibatch_defined_by_labels(tmpdir):

    input_dim = 1000
    num_output_classes = 5

    def assert_data(mb_source):
        features_si = mb_source.stream_info('features')
        labels_si = mb_source.stream_info('labels')

        mb = mb_source.next_minibatch(2)

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
                               [[1.,  0.,  0.,  0.,  0.]],
                               [[0.,  1.,  0.,  0.,  0.]]
                           ]))

        mb = mb_source.next_minibatch(3)
        features = mb[features_si]
        labels = mb[labels_si]

        assert features.num_samples == 10
        assert labels.num_samples == 3

    tmpfile = _write_data(tmpdir, MBDATA_SPARSE)
    mb_source = MinibatchSource(CTFDeserializer(tmpfile, StreamDefs(
        features=StreamDef(field='x', shape=input_dim, is_sparse=True),
        labels=StreamDef(field='y', shape=num_output_classes, is_sparse=False, defines_mb_size=True)
    )), randomize=False)

    assert_data(mb_source)

    tmpfile1 = _write_data(tmpdir, MBDATA_SPARSE1, '1')
    tmpfile2 = _write_data(tmpdir, MBDATA_SPARSE2, '2')
    combined_mb_source = MinibatchSource([ CTFDeserializer(tmpfile1, StreamDefs(
            features=StreamDef(field='x', shape=input_dim, is_sparse=True))),
        CTFDeserializer(tmpfile2, StreamDefs(
            labels=StreamDef(field='y', shape=num_output_classes, is_sparse=False, defines_mb_size=True)
        ))], randomize=False)

    assert_data(combined_mb_source)


# Create base64 and usual image deserializers
# and check that they give equal minibatch data on
# the same input images
def test_base64_is_equal_image(tmpdir):
    import io, base64; from PIL import Image
    np.random.seed(1)

    file_mapping_path = str(tmpdir / 'file_mapping.txt')
    base64_mapping_path = str(tmpdir / 'base64_mapping.txt')

    with open(file_mapping_path, 'w') as file_mapping:
        with open(base64_mapping_path, 'w') as base64_mapping:
            for i in range(10):
                data = np.random.randint(0, 2**8, (5,7,3))
                image = Image.fromarray(data.astype('uint8'), "RGB")
                buf = io.BytesIO()
                image.save(buf, format='PNG')
                assert image.width == 7 and image.height == 5
                
                label = str(i) 
                # save to base 64 mapping file
                encoded = base64.b64encode(buf.getvalue()).decode('ascii')
                base64_mapping.write('%s\t%s\n' % (label, encoded))
         
                # save to mapping + png file
                file_name = label + '.png'
                with open(str(tmpdir/file_name), 'wb') as f:
                    f.write(buf.getvalue())
                file_mapping.write('.../%s\t%s\n' % (file_name, label))

    transforms = [xforms.scale(width=7, height=5, channels=3)]
    b64_deserializer = Base64ImageDeserializer(base64_mapping_path,
        StreamDefs(
            images1=StreamDef(field='image', transforms=transforms),
            labels1=StreamDef(field='label', shape=10)))

    file_image_deserializer = ImageDeserializer(file_mapping_path,
        StreamDefs(
            images2=StreamDef(field='image', transforms=transforms),
            labels2=StreamDef(field='label', shape=10)))

    mb_source = MinibatchSource([b64_deserializer, file_image_deserializer])
    for j in range(20):
        mb = mb_source.next_minibatch(1)

        images1_stream = mb_source.streams['images1']
        images1 = mb[images1_stream].asarray()
        images2_stream = mb_source.streams['images2']
        images2 = mb[images2_stream].asarray()
        assert(images1 == images2).all()

def test_crop_dimensionality(tmpdir):
    import io; from PIL import Image
    np.random.seed(1)

    file_mapping_path = str(tmpdir / 'file_mapping.txt')
    with open(file_mapping_path, 'w') as file_mapping:
        for i in range(5):
            data = np.random.randint(0, 2**8, (20, 40, 3))
            image = Image.fromarray(data.astype('uint8'), "RGB")
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            assert image.width == 40 and image.height == 20
            
            label = str(i) 
            # save to mapping + png file
            file_name = label + '.png'
            with open(str(tmpdir/file_name), 'wb') as f:
                f.write(buf.getvalue())
            file_mapping.write('.../%s\t%s\n' % (file_name, label))

    transforms1 = [
        xforms.scale(width=40, height=20, channels=3),
        xforms.crop(crop_type='randomside', 
                    crop_size=(20, 10), side_ratio=(0.2, 0.5),
                    jitter_type='uniratio')]

    transforms2 = [
        xforms.crop(crop_type='randomside', 
                    crop_size=(20, 10), side_ratio=(0.2, 0.5),
                    jitter_type='uniratio')]

    d1 = ImageDeserializer(file_mapping_path,
        StreamDefs(
            images1=StreamDef(field='image', transforms=transforms1),
            labels1=StreamDef(field='label', shape=10)))

    d2 = ImageDeserializer(file_mapping_path,
        StreamDefs(
            images2=StreamDef(field='image', transforms=transforms2),
            labels2=StreamDef(field='label', shape=10)))

    mbs = MinibatchSource([d1, d2])
    for j in range(5):
        mb = mbs.next_minibatch(1)
        images1 = mb[mbs.streams.images1].asarray()
        images2 = mb[mbs.streams.images2].asarray()
        assert images1.shape == (1, 1, 3, 10, 20)
        assert (images1 == images2).all()

def test_prefetch_with_unpacking(tmpdir):
    data = r'''0  |S0 1 1 1 1   |S1 1000
1   |S0 2 2 2 2  |S1 100
2   |S0 3 3 3 3  |S1 100
3   |S0 1 1 1 1  |S1 10
4   |S0 2 2 2 2  |S1 1
5   |S0 3 3 3 3  |S1 2000
6   |S0 1 1 1 1  |S1 200
7   |S0 2 2 2 2  |S1 200
8   |S0 3 3 3 3  |S1 20
9   |S0 1 1 1 1  |S1 2
'''
    import time
    tmpfile = _write_data(tmpdir, data)

    input_dim = 4
    num_output_classes = 1

    mb_source = MinibatchSource(CTFDeserializer(tmpfile, StreamDefs(
        features=StreamDef(field='S0', shape=input_dim, is_sparse=False),
        labels=StreamDef(field='S1', shape=num_output_classes, is_sparse=False)
    )), randomize=False, max_samples=FULL_DATA_SWEEP)

    input_map = { 'S0' : mb_source.streams.features, 'S1' : mb_source.streams.labels }
    empty = False
    mb_size = 3
    # On the last minibatch there will be resize called, 
    # due to 10%3 = 1 sample  in the minibatch
    while not empty:
        mb = mb_source.next_minibatch(mb_size, input_map=input_map)
        time.sleep(1) # make sure the prefetch kicks in
        if mb:
            # Force unpacking to check that we do 
            # not break prefetch 
            actual_size = mb['S0'].shape[0]
            assert (mb['S0'].asarray() == np.array([[[1, 1, 1, 1]],
                                                    [[2, 2, 2, 2]],
                                                    [[3, 3, 3, 3]]], dtype=np.float32)[0:actual_size]).all()
        else:
            empty = True

def get_cbf_header(streams):
    get_header_line = lambda x,y: \
        [x, y.stream_alias, 'sparse' if y.is_sparse else 'dense', str(y.dim)]
    return [' '.join(get_header_line(k,v)) for k,v in streams.items()]

input_files =  [
        MBDATA_DENSE_1, 
        MBDATA_DENSE_2, 
        MBDATA_SPARSE, 
        MBDATA_SPARSE1, 
        MBDATA_SPARSE2
    ]
stream_defs = [
        StreamDefs(
            features=StreamDef(field='S0', shape=1),
            labels=StreamDef(field='S1', shape=1)
        ),
        StreamDefs(
            features=StreamDef(field='S0', shape=1),
            labels=StreamDef(field='S1', shape=1)
        ),
        StreamDefs(
            features=StreamDef(field='x', shape=1000, is_sparse=True),
            labels=StreamDef(field='y', shape=5, is_sparse=False)
        ),
        StreamDefs(
            features=StreamDef(field='x', shape=1000, is_sparse=True),
        ),
        StreamDefs(
            labels=StreamDef(field='y', shape=5, is_sparse=False)
        ),
    ]

@pytest.mark.parametrize("input_pair", zip(input_files, stream_defs))
def test_compare_cbf_and_ctf(input_pair, device_id, tmpdir):
    try:
        import ctf2bin
    except ImportError:
        pytest.skip("ctf2bin not found")

    device = cntk_device(device_id)

    tmpfile = _write_data(tmpdir, input_pair[0])
    streams = input_pair[1]

    ctf2bin.process(tmpfile, tmpfile+'.bin', get_cbf_header(streams), ctf2bin.ElementType.FLOAT)

    def compare_cbf_and_ctf(num_mbs, mb_size, randomize):
        ctf = MinibatchSource(CTFDeserializer(tmpfile, streams), randomize=randomize)
        cbf = MinibatchSource(CBFDeserializer(tmpfile+'.bin', streams), randomize=randomize)

        ctf_stream_names = sorted([x.m_name for x in ctf.stream_infos()])
        cbf_stream_names = sorted([x.m_name for x in cbf.stream_infos()])

        assert(ctf_stream_names == cbf_stream_names)
        for _ in range(num_mbs):
            ctf_mb = ctf.next_minibatch(mb_size, device=device)
            cbf_mb = cbf.next_minibatch(mb_size, device=device)

            for name in cbf_stream_names:
                ctf_data = ctf_mb[ctf[name]]
                cbf_data = cbf_mb[cbf[name]]

                
                assert ctf_data.num_samples == cbf_data.num_samples
                assert ctf_data.num_sequences == cbf_data.num_sequences
                assert ctf_data.shape == cbf_data.shape
                assert ctf_data.end_of_sweep == cbf_data.end_of_sweep
                assert ctf_data.is_sparse == cbf_data.is_sparse
                assert ctf_data.data.masked_count() == cbf_data.data.masked_count()

                # XXX:
                # assert(ctf_data.asarray() == cbf_data.asarray()).all()
                # not using asarray because for sparse values it fails with
                # some strange exception "sum of the rank of the mask and Variable 
                #rank does not equal the Value's rank".

                assert C.cntk_py.are_equal(ctf_data.data.data, cbf_data.data.data)

                if (ctf_data.data.masked_count() > 0):
                    assert (ctf_data.data.mask == cbf_data.data.mask).all()
                # XXX: if mask_count is zero, mb_data.data.mask fails with 
                # "AttributeError: 'Value' object has no attribute 'mask'"!

                # XXX: without invoking erase, next_minibatch will fail with:
                # "Resize: Cannot resize the matrix because it is a view."
                ctf_data.data.erase()
                cbf_data.data.erase()

    for randomize in [False, True]:
        for (num_mbs, mb_size) in zip([1, 1, 3, 10], [1, 10, 100, 2]):
            compare_cbf_and_ctf(num_mbs, mb_size, randomize)

# Helper generator
class GenDeserializer(UserDeserializer):
    def __init__(self, stream_infos, num_chunks, num_sequences, max_sequence_len = 1):
        super(GenDeserializer, self).__init__()
        self._streams = stream_infos
        self._num_chunks = num_chunks
        self._num_sequences = num_sequences
        self._max_sequence_len = max_sequence_len

    def stream_infos(self):
        return self._streams;

    def num_chunks(self):
        return self._num_chunks

    def get_chunk(self, chunk_id):
        import scipy.sparse as sp, random, functools
        random.seed(chunk_id)
        result = {}
        for stream in self._streams:
            total = functools.reduce(lambda x, y: x*y, stream.sample_shape)
            count = 0       
            chunk = []
            for i in range(self._num_sequences):
                if self._max_sequence_len == 1:
                    shape = stream.sample_shape
                else:
                    seq_len = random.randint(1, self._max_sequence_len)
                    shape = [seq_len]
                    shape.extend(stream.sample_shape)
                    shape = tuple(shape) if stream.storage_format == 'dense' else (shape[0], total)

                if stream.storage_format == 'dense':
                    data = np.full(shape=shape, fill_value=chunk_id, dtype=np.float32)
                else:
                    data = np.full(shape=shape, fill_value=chunk_id, dtype=np.float32)
                    data = sp.csr_matrix(data, shape=shape, dtype=np.float32)
                chunk.append(data)

            if stream.storage_format == 'dense':
                result[stream.name] = chunk if self._max_sequence_len != 1 else np.stack(chunk)
            else: # sparse
                result[stream.name] = chunk if self._max_sequence_len != 1 else sp.csr_matrix(sp.vstack(chunk))
        return result

def test_user_deserializer_sample_mode():
    import scipy.sparse as sp
    streams = [StreamInformation('x', 0, 'dense', np.float32, (2, 3)), 
               StreamInformation('y', 1, 'sparse', np.float32, (1, 3))]

    def run_minibatch_source(minibatch_source, num_chunks, num_samples_per_value):
        sample_x_values = np.zeros(num_chunks, dtype=np.int32)
        sample_y_values = np.zeros(num_chunks, dtype=np.int32)
        mb_count = 0
        while True:
            if mb_count % 10 == 1: # perform checkpointing
                checkpoint_state = minibatch_source.get_checkpoint_state()
                for i in range(3): 
                    minibatch_source.next_minibatch(20)
                minibatch_source.restore_from_checkpoint(checkpoint_state)
                mb_count +=1
                continue            
            mb = minibatch_source.next_minibatch(20)
            mb_count += 1
            if not mb:
                break

            for sequence in mb[minibatch_source.streams.x].asarray():
                for sample in sequence:
                    value = int(sample[0][0])
                    sample_x_values[value] += 1

            for sequence in mb[minibatch_source.streams.y].asarray():
                for sample in sequence:
                    value = int(sample[0][0])
                    sample_y_values[value] += 1
            mb = None

        expected_values = np.full(num_chunks, fill_value=num_samples_per_value, dtype=np.int32)
        assert (sample_x_values == expected_values).all()
        assert (sample_y_values == expected_values).all()

    # Big chunks
    d = GenDeserializer(stream_infos=streams, num_chunks=20, num_sequences=100)
    mbs = MinibatchSource([d], randomize=False, max_sweeps=2)
    run_minibatch_source(mbs, num_chunks=20, num_samples_per_value=200)
    # Randomized
    mbs = MinibatchSource([d], randomize=True, max_sweeps=2, randomization_window_in_chunks=5)
    run_minibatch_source(mbs, num_chunks=20, num_samples_per_value=200)

    # Small chunks of 1
    d = GenDeserializer(stream_infos=streams, num_chunks=20, num_sequences=1)
    mbs = MinibatchSource([d], randomize=False, max_sweeps=3)
    run_minibatch_source(mbs, num_chunks=20, num_samples_per_value=3)
    # Randomized
    mbs = MinibatchSource([d], randomize=True, max_sweeps=3, randomization_window_in_chunks=5)
    run_minibatch_source(mbs, num_chunks=20, num_samples_per_value=3)

def test_user_deserializer_sequence_mode():
    import scipy.sparse as sp
    streams = [StreamInformation('x', 0, 'dense', np.float32, (2, 3)), 
               StreamInformation('y', 1, 'sparse', np.float32, (3,))]

    def run_minibatch_source(minibatch_source, num_chunks, num_sequences_per_value):
        sequence_x_values = np.zeros(num_chunks, dtype=np.int32)
        sequence_y_values = np.zeros(num_chunks, dtype=np.int32)
        mb_count = 0
        while True:
            if mb_count % 10 == 1: # perform checkpointing
                checkpoint_state = minibatch_source.get_checkpoint_state()
                for i in range(3): 
                    minibatch_source.next_minibatch(20)
                minibatch_source.restore_from_checkpoint(checkpoint_state)
                mb_count +=1
                continue            

            mb = minibatch_source.next_minibatch(20)
            mb_count += 1
            if not mb:
                break

            for sequence in mb[minibatch_source.streams.x].asarray():
                sequence_x_values[int(sequence[0][0][0])] +=1

            for sequence in mb[minibatch_source.streams.y].as_sequences(C.sequence.input_variable((3,), True)):             
                sequence_y_values[int(sequence.toarray()[0][0])] += 1
            mb = None

        expected_values = np.full(num_chunks, fill_value=num_sequences_per_value, dtype=np.int32)
        assert (sequence_x_values == expected_values).all()
        assert (sequence_y_values == expected_values).all()

    # Big chunks
    d = GenDeserializer(stream_infos=streams, num_chunks=15, 
                        num_sequences=100, max_sequence_len=10)
    mbs = MinibatchSource([d], randomize=False, max_sweeps=2)
    run_minibatch_source(mbs, num_chunks=15, num_sequences_per_value=200)
    # Randomized
    mbs = MinibatchSource([d], randomize=True, max_sweeps=2, randomization_window_in_chunks=5)
    run_minibatch_source(mbs, num_chunks=15, num_sequences_per_value=200)

    # Small chunks of 1
    d = GenDeserializer(stream_infos=streams, num_chunks=15,
                        num_sequences=1, max_sequence_len=10)
    mbs = MinibatchSource([d], randomize=False, max_sweeps=3)
    run_minibatch_source(mbs, num_chunks=15, num_sequences_per_value=3)
    # Randomized
    mbs = MinibatchSource([d], randomize=True, max_sweeps=3, randomization_window_in_chunks=5)
    run_minibatch_source(mbs, num_chunks=15, num_sequences_per_value=3)

def test_index_caching(tmpdir):
    pytest.skip("test_index_caching is disabled")
    import os, time, glob, uuid
    MB = 1 << 20
    data = MBDATA_DENSE_1
    while(len(data) < 64 * MB):
        data += data

    timeWithoutCache, timeWithCache = 0, 0 

    cpu=C.device.cpu()
    streams = stream_defs[0]

    for _ in range(3):
        tmpfile = _write_data(tmpdir, data, str(uuid.uuid4()))

        cache_files = glob.glob(str(tmpdir + '/*.cache'))
        for cache_file in cache_files:
            os.remove(cache_file)

        config = CTFDeserializer(tmpfile, streams)
        config['cacheIndex'] = C.cntk_py.DictionaryValue(True)

        start = time.time()
        MinibatchSource(config, randomize=False).next_minibatch(1, device=cpu)
        end = time.time()

        timeWithoutCache += (end - start)

        time.sleep(5)
        
        cache_files = glob.glob(str(tmpdir + '/*.cache'))
        assert len(cache_files) == 1


        start = time.time()
        MinibatchSource(config, randomize=False).next_minibatch(1, device=cpu)
        end = time.time()

        os.remove(tmpfile)

        timeWithCache += (end - start)

    assert timeWithCache < timeWithoutCache
