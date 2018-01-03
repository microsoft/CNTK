import cntk as C
import util as util
import time
from cntk.device import try_set_default_device, cpu
import numpy as np
import sys, os
try_set_default_device(cpu())

def get_initializer(init_method, params):
    if init_method == 'tnormal':
        return C.initializer.truncated_normal(stdev=params['init_value'])
    elif init_method == 'uniform':
        return C.initializer.uniform(scale=params['init_value'])
    elif init_method == 'normal':
        return C.initializer.normal(scale=params['init_value'])
    else:
        raise ValueError('initial method {0} is not support in CNTK'.format(init_method))


def active_layer(func_name, x):
    if func_name == 'relu':
        return C.relu(x)
    elif func_name == 'sigmid':
        return C.sigmoid(x)
    elif func_name == 'tanh':
        return C.tanh(x)
    else:
        raise ValueError('activaticon func {0} is not supported in CNTK'.format(func_name))


def print_training_process(trainer, mb, frequency, verbose=1):
    loss, logloss = "NA", "NA"
    if mb % frequency == 0:
        loss = trainer.previous_minibatch_loss_average
        logloss = trainer.previous_minibatch_evaluation_average
        if verbose:
            print("minibatch: {0}, loss: {1:.4f}, logloss: {2:.4f}".format(mb, loss, logloss))
    return mb, loss, logloss


def linear_part(lr_input, input_dim, model_param, initializer):
    weight = C.parameter(shape=(input_dim, 1), init=initializer)
    bias = C.parameter(shape=(1))
    lr_out = bias + C.times(lr_input, weight)
    return lr_out


def nn_part(embedding, dnn_input, model_param, params, initializer):
    feature_cnt = params['feature_cnt']
    embedding_dim = params['embedding_dim']
    field_cnt = params['field_cnt']
    dnn_input = C.reshape(C.unpack_batch(dnn_input), (-1, feature_cnt))
    nn_input = C.reshape(C.times(dnn_input, embedding), (-1, field_cnt * embedding_dim))
    layer_sizes = params['layer_sizes']
    layer_activations = params['layer_activations']
    hidden_nn_layers = []
    hidden_nn_layers.append(nn_input)
    last_layer_size = field_cnt * embedding_dim
    for idx, layer_size in enumerate(layer_sizes):
        curr_w_nn_layer = C.parameter(shape=(last_layer_size, layer_size), init=initializer)
        curr_b_nn_layer = C.parameter(shape=(layer_size), init=initializer)
        curr_hidden_nn_layer = active_layer(layer_activations[idx],
                                            curr_b_nn_layer + C.times(hidden_nn_layers[idx], curr_w_nn_layer))
        last_layer_size = layer_size
        hidden_nn_layers.append(curr_hidden_nn_layer)
        model_param.append(curr_w_nn_layer)
        model_param.append(curr_b_nn_layer)
    w_nn_output = C.parameter(shape=(last_layer_size, 1), init=initializer)
    b_nn_output = C.parameter(shape=(1), init=initializer)
    model_param.append(w_nn_output)
    model_param.append(b_nn_output)
    nn_out = C.times(hidden_nn_layers[-1], w_nn_output) + b_nn_output
    return nn_out


def buildNetWork(lr_input, dnn_input, label, params):
    model_param = []
    feature_cnt = params['feature_cnt']
    embedding_dim = params['embedding_dim']
    init_method = params['init_method']
    layer_l2 = params['layer_l2']
    embed_l2 = params['embed_l2']
    global_initializer = get_initializer(init_method, params)
    embedding = C.parameter(shape=(feature_cnt, embedding_dim), init=global_initializer)
    lr_out = linear_part(lr_input, feature_cnt, model_param, global_initializer)
    nn_out = nn_part(embedding, dnn_input, model_param, params, global_initializer)
    pred = lr_out + nn_out
    out = C.sigmoid(pred)

    w_norm_layer = C.Constant(0)
    for index, p in enumerate(model_param):
        w_norm_layer = C.plus(w_norm_layer, 0.5 * C.reduce_mean(C.square(p)))
    w_norm_embed = C.Constant(0)
    w_norm_embed = C.plus(w_norm_embed, 0.5 * C.reduce_mean(C.square(embedding)))
    loss = C.reduce_mean(C.binary_cross_entropy(out, label)) + layer_l2 * w_norm_layer + embed_l2 * w_norm_embed
    logloss = C.reduce_mean(C.binary_cross_entropy(out, label))
    return out, loss, logloss


def eval(lr_input, dnn_input, label, trainer, file_name):
    eval_aggregate_loss = 0.0
    eval_sample_num = 0

    for training_input_in_sp in util.load_data_cache(file_name):
        logloss_res = trainer.test_minibatch(
            {lr_input: training_input_in_sp['lr_input'], label: training_input_in_sp['labels'],
             dnn_input: training_input_in_sp['dnn_input']})
        eval_sample_num += training_input_in_sp['lr_input'].shape[0]
        eval_aggregate_loss += logloss_res * training_input_in_sp['lr_input'].shape[0]
    eval_logloss = eval_aggregate_loss / eval_sample_num

    return eval_logloss


def train(params):
    learning_rate = params['learning_rate']
    n_epochs = params['n_epochs']
    show_steps = params['show_steps']
    batch_size = params['batch_size']
    feature_cnt = params['feature_cnt']
    field_cnt = params['field_cnt']

    # define model
    lr_input = C.input_variable((batch_size, feature_cnt), is_sparse=True)
    dnn_input = C.input_variable((batch_size, field_cnt * feature_cnt), is_sparse=True)
    label = C.input_variable((batch_size, 1), np.float32)
    pred, loss, logloss = buildNetWork(lr_input, dnn_input, label, params)

    lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
    learner = C.momentum_sgd(pred.parameters, lr_schedule, momentum=0.9, use_mean_gradient=True)
    trainer = C.Trainer(pred, (loss, logloss), [learner])

    i = 0
    start = time.time()
    log_writer = open('./log.txt','w')
    final_train_logloss = 0.0
    for epoch in range(n_epochs):
        aggregate_loss = 0.0
        for training_input_in_sp in util.load_data_cache(params['train_cache']):
            i += 1
            trainer.train_minibatch({lr_input: training_input_in_sp['lr_input'], label: training_input_in_sp['labels'],
                                     dnn_input: training_input_in_sp['dnn_input']})
            print_training_process(trainer, i, show_steps, verbose=1)
        train_info = eval(lr_input, dnn_input, label, trainer, params['train_cache'])
        final_train_logloss = train_info
        eval_info = eval(lr_input, dnn_input, label, trainer, params['eval_cache'])
        print('epoch: {0}, train logloss: {1:.4f}, eval logloss: {2:.4f}'.format(epoch, train_info, eval_info))
        log_writer.write('epoch: {0}, train logloss: {1:.4f}, eval logloss: {2:.4f}'.format(epoch, train_info, eval_info)+'\n')
    log_writer.close()
    end = time.time()
    print('total used time:', end - start)
    return final_train_logloss


def deepWide():
    abs_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(abs_path, "..", "data", "deepWide")
    # parameter setting
    params = {
        'feature_cnt': 194081,
        'field_cnt': 33,
        'batch_size': 3,
        'train_file': '/train.entertainment.no_inter.norm.fieldwise.userid.txt',
        'eval_file': '/val.entertainment.no_inter.norm.fieldwise.userid.txt',
        'embedding_dim': 10,
        'layer_l2': 0.1,
        'embed_l2': 0.1,
        'init_value': 0.001,
        'init_method': 'normal',
        'n_epochs': 5,
        'learning_rate': 0.01,
        'show_steps': 20,
        'layer_sizes': [50],
        'layer_activations': ['relu']
    }
    params['train_file'] = data_path + params['train_file']
    params['eval_file'] = data_path + params['eval_file']
    # cache train data and eval data
    params['train_cache'] = params['train_file'].replace('.txt', '.pkl')
    params['eval_cache'] = params['eval_file'].replace('.txt', '.pkl')
    util.pre_build_data_cache(params['train_file'], params['train_cache'], params)
    util.pre_build_data_cache(params['eval_file'], params['eval_cache'], params)
    final_train_loglss = train(params)
    return final_train_loglss

if __name__ == "__main__":
    final_train_loglss = deepWide()