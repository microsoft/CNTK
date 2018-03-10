import pytest
import numpy as np
import cntk as C

TEST_CONFIG = [
    # (num_layers, bidirectional, recurrent_op
    (1, True,  'lstm'),
    (1, False, 'lstm'),
    (2, False, 'lstm'),
    (3, True,  'lstm'),
    (4, True,  'rnnReLU'),
    (4, False, 'rnnTanh'),
]

@pytest.mark.parametrize("num_layers, bidirectional, recurrent_op", TEST_CONFIG)
def test_convert_optimized_rnnstack(num_layers, bidirectional, recurrent_op, device_id):
    if device_id == -1:
        pytest.skip('only runs on GPU')

    input_dim = 5
    hidden_dim = 3
    batches = [[np.random.random((20,input_dim)).astype(np.float32), np.random.random((10,input_dim)).astype(np.float32), np.random.random((40,input_dim)).astype(np.float32)],
               [np.random.random((1,input_dim)).astype(np.float32)]]
    
    for data in batches:
        input_var = C.sequence.input_variable(shape=(input_dim,))
        
        W1 = C.parameter((-1,1), init = C.glorot_uniform())
        W2 = C.parameter((-1,1), init = C.glorot_uniform())
        cudnn_rnn1 = C.optimized_rnnstack(input_var, W1, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, recurrent_op=recurrent_op)
        dense1 = C.layers.Dense(hidden_dim)(cudnn_rnn1)
        cudnn_rnn2 = C.optimized_rnnstack(dense1, W2, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, recurrent_op=recurrent_op)
        dense2 = C.layers.Dense(hidden_dim)(cudnn_rnn2)
        cudnn_rnn3 = C.optimized_rnnstack(dense2, W2, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, recurrent_op=recurrent_op) # test shared parameter W2
        
        def blocked(d):
            blocked_W = C.parameter((-1,d), init = C.glorot_uniform())
            @C.layers.BlockFunction('', '')
            def func(x):
                return C.optimized_rnnstack(x, blocked_W, d, 1, recurrent_op='lstm')
            return func
        
        cudnn_model = C.layers.Sequential([blocked(hidden_dim), blocked(2*hidden_dim), blocked(3*hidden_dim)])(cudnn_rnn3)
        cudnn_out = cudnn_model.eval({input_var:data})

        model = C.misc.convert_optimized_rnnstack(cudnn_model)

        # make sure original cudnn model is intact
        cudnn_out2 = cudnn_model.eval({input_var:data})
        assert all(np.allclose(cudnn_out[i], cudnn_out2[i]) for i in range(len(cudnn_out)))

        model_out = model.eval({model.arguments[0]:data})
        assert all(np.allclose(cudnn_out[i], model_out[i]) for i in range(len(cudnn_out)))