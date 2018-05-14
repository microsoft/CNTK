import numpy as np
import cntk as C
import pytest


from functools import partial

from .. import stream_infos_from_variables, next_chunk_function_from_nextminbatch_function, compose_chunk_function,\
    LambdaDeserializer, LambdaSequentialDeserializer,chunk_function_from_index_data_function, chunk_function_from_index_batch_function

def test_labmda_deserilizer():
    N_Chunk = 3
    N = 10
    N_seq = 3

    from scipy import sparse as sp

    X = np.arange(3 * N * N_Chunk).reshape(N_Chunk, N, 3).astype(np.float32)  # 5 rows of 3 values
    Y = (np.arange(3 * N).reshape(N, 3).astype(np.float32))

    x_func = lambda ch_id, data: data[ch_id % N_Chunk]
    y_func = lambda ch_id: np.arange(4 * N).reshape(N, 4).astype(np.float32)  # [d for d in Y]
    # z_func = lambda ch_id:  [d for d in np.ones((N, N_seq, 3, 1)).astype(np.float32)]
    # def z_func(ch_id):
    #     return Y #np.copy(np.ones((N, 3, 1)).astype(np.float32))
    # z_func = lambda ch_id: np.random.rand(N, 3, 1).astype(np.float32)
    # def z_func(ch_id):
    #     return np.random.rand(N, 3, 1).astype(np.float32)
    x_func_real = partial(x_func, data=X)
    meta_infos = [dict(name='X', is_sparse=False, dtype=np.float32, shape=x_func_real(0).shape[1:]),
                  dict(name='Y', is_sparse=False, dtype=np.float32, shape=y_func(0).shape[1:]),
                  # dict(name='Z', is_sparse=False, dtype=np.float32, shape=z_func(0)[0].shape[1:])
                  ]

    data_fs = {'X': x_func_real, 'Y': y_func}  # , 'Z': partial(z_func)}

    def next_minibatch_func(N):
        seq_size = 5
        return {'X': np.arange(3 * N).reshape(N, 3).astype(np.float32),
                'Y': np.arange(3 * N).reshape(N, 3).astype(np.float32) + 100,
                'Z': [np.arange(3 * seq_size).reshape(seq_size, 3, 1).astype(np.float32) for _ in range(N)]
                }

    x = C.input_variable(3, name='X')
    y = C.input_variable(4, name='Y')
    z = C.sequence.input_variable((3, 1), name='Z')
    meta_infos = stream_infos_from_variables([x, y, z])

    chunk_func_from_mb_function = next_chunk_function_from_nextminbatch_function(next_minibatch_func, 5, 2)
    print('chunk function from next_minibatch_f: ', chunk_func_from_mb_function)
    print('chunk function from next_minibatch_f next(): ', chunk_func_from_mb_function())
    print('chunk function from next_minibatch_f next(): ', chunk_func_from_mb_function())

    chunk_func = compose_chunk_function(data_fs)
    print('compose chunk function: ', chunk_func)
    print('compose chunk function output [0]: ', chunk_func(0))
    print('compose chunk function output [1]: ', chunk_func(1))

    def init_chunk_func():
        chunk_func = next_chunk_function_from_nextminbatch_function(next_minibatch_func, 5, 2)
        return chunk_func

    print('num chunks: ', LambdaSequentialDeserializer(meta_infos, init_chunk_func).num_chunks())
    mbs = C.io.MinibatchSource([LambdaSequentialDeserializer(meta_infos, init_chunk_func)], randomize=False)
    for i in range(2):
        mb = mbs.next_minibatch(4)
        result_x = mb[mbs.streams['X']].data.asarray()
        result_y = mb[mbs.streams['Y']].data.asarray()
        result_z = mb[mbs.streams['Z']].data.asarray()

    print(result_x)
    print(result_y)
    print(result_z)

    input_map = {x: mbs['X'], y: mbs['Y'], z: mbs['Z']}  # , z: mbs['Z']}
    res = (C.reduce_sum(x, axis=-1) * y * z).eval(mbs.next_minibatch(16, input_map))

    print('res:', res)
    print('len: ', len(res))
    print(res[0].shape)
    print([d for d in res[0]])

    def index_data_func(i):
        seq_size = 5

        return {'X': np.arange(3).reshape(3).astype(np.float32),
                'Y': np.arange(3).reshape(3).astype(np.float32) + 100,
                'Z': [np.arange(3 * seq_size).reshape(seq_size, 3, 1).astype(np.float32)]
                }
    chunk_func = chunk_function_from_index_data_function(index_data_func, 5, 10)
    print('chunk[0]', chunk_func(0))
    print('chunk[1]', chunk_func(1))
    mbs = C.io.MinibatchSource([LambdaDeserializer(meta_infos, chunk_func, num_chunk=5)], randomize=False)
    for i in range(2):
        mb = mbs.next_minibatch(4)
        result_x = mb[mbs.streams['X']].data.asarray()
        result_y = mb[mbs.streams['Y']].data.asarray()
        result_z = mb[mbs.streams['Z']].data.asarray()

    def index_batch_func(i):
        seq_size = 5

        return {'X': np.arange(3 * N).reshape(N, 3).astype(np.float32),
                'Y': np.arange(3 * N).reshape(N, 3).astype(np.float32) + 100,
                'Z': [np.arange(3 * seq_size).reshape(seq_size, 3, 1).astype(np.float32) for _ in range(N)]
                }

    chunk_func = chunk_function_from_index_batch_function(index_batch_func, 5, 15)
    print('chunk[0]', chunk_func(0))
    print('chunk[1]', chunk_func(1))
    mbs = C.io.MinibatchSource([LambdaDeserializer(meta_infos, chunk_func, num_chunk=5)], randomize=False)
    for i in range(2):
        mb = mbs.next_minibatch(4)
        result_x = mb[mbs.streams['X']].data.asarray()
        result_y = mb[mbs.streams['Y']].data.asarray()
        result_z = mb[mbs.streams['Z']].data.asarray()