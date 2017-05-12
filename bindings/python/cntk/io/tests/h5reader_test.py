import h5py
import numpy as np
import pytest

import cntk as C
from cntk.io.h5reader import H5DataSource

def test_text_format(tmpdir):
    tmph5 = str(tmpdir / 'mbdata.h5')
    with h5py.File(tmph5, "w") as f:
        # sequence lenth is fixed to 3; element shape is (4, )
        num_data_point = 100
        fixed_len_seq = f.create_dataset("fixed_len_seq", (num_data_point, 3, 4), dtype=np.float32)
        fixed_len_seq.attrs['is_seq'] = True
        fixed_len_seq.attrs['shape'] = (4,)
        fixed_len_seq.attrs['dtype'] = 'f4'  # float32

        vlen_seq = f.create_dataset("vlen_seq",
                                    (num_data_point,),
                                    h5py.special_dtype(vlen=np.dtype('f4'))
                                    # or using:
                                    # h5py.special_dtype(vlen=np.float32)
                                    )
        vlen_seq.attrs['is_seq'] = True
        vlen_seq.attrs['shape'] = (3,)
        vlen_seq.attrs['dtype'] = 'f4'  # float32

        label = f.create_dataset("label", (num_data_point, 1), dtype=np.float32)

        # create facke data
        for i in range(num_data_point):
            fixed_len_seq[i] = np.ones((3, 4)) * i
            vlen_seq_inshape = np.ones(((i % 4) + 1, 3)) * i

            vlen_seq[i] = np.reshape(vlen_seq_inshape, np.prod(vlen_seq_inshape.shape))
            label[i] = np.array([i])

        # use the dataset with H5DataSource
        # ds = H5DataSource(f)
        ds = H5DataSource({'fixed_len_seq': f['fixed_len_seq'], 'label': f['label'], 'vlen_seq': f['vlen_seq']})

        # try minibatch
        def eval_batch(batch):
            v_fixed_seq = C.sequence.input(4)
            fixed_seq_op_result = (C.sequence.last(v_fixed_seq) * 1).eval(
                {v_fixed_seq: batch[ds.stream_info_mapping['fixed_len_seq']]})
            v_vlen_seq = C.sequence.input(3)
            v_vlen_seq_op_result = (C.sequence.last(v_vlen_seq) * 1).eval(
                {v_vlen_seq: batch[ds.stream_info_mapping['vlen_seq']]})
            v_label = C.input(1)
            v_label_op_result = (v_label * 1).eval({v_label: batch[ds.stream_info_mapping['label']]})
            return fixed_seq_op_result, v_vlen_seq_op_result, v_label_op_result


        seq_len = 3
        num_seq = 3
        batch_size = num_seq * seq_len
        batch = ds.next_minibatch(batch_size)
        fixed_seq_op_result, v_vlen_seq_op_result, v_label_op_result = eval_batch(batch)

        assert np.allclose(fixed_seq_op_result,
                           np.asarray([[0., 0., 0., 0.],
                                       [1., 1., 1., 1.],
                                       [2., 2., 2., 2.]]))
        assert np.allclose(v_vlen_seq_op_result,
                           np.asarray([[0., 0., 0.],
                                       [1., 1., 1.],
                                       [2., 2., 2.]]))
        assert np.allclose(v_label_op_result,
                           np.asarray([
                               [0.],
                               [1.],
                               [2.]
                           ]))
        #next batch
        batch = ds.next_minibatch(batch_size)
        fixed_seq_op_result, v_vlen_seq_op_result, v_label_op_result = eval_batch(batch)
        assert np.allclose(fixed_seq_op_result,
                           np.asarray([[3., 3., 3., 3.],
                                       [4., 4., 4., 4.],
                                       [5., 5., 5., 5.]]))
        assert np.allclose(v_vlen_seq_op_result,
                           np.asarray([[3., 3., 3.],
                                       [4., 4., 4.],
                                       [5., 5., 5.]]))
        assert np.allclose(v_label_op_result,
                           np.asarray([
                               [3.],
                               [4.],
                               [5.]
                           ]))
        #cycle back to the starting point
        for i in range(32):
            batch = ds.next_minibatch(batch_size)
        fixed_seq_op_result, v_vlen_seq_op_result, v_label_op_result = eval_batch(batch)
        assert np.allclose(fixed_seq_op_result,
                           np.asarray([[99., 99., 99., 99.],
                                       [0., 0., 0., 0.],
                                       [1., 1., 1., 1.]]))
        assert np.allclose(v_vlen_seq_op_result,
                           np.asarray([[99., 99., 99.],
                                       [0., 0., 0.],
                                       [1., 1., 1.] ]))
        assert np.allclose(v_label_op_result,
                           np.asarray([
                               [99.],
                               [0.],
                               [1.]
                           ]))

        f.close()