import cntk as C
import numpy as np
from polymath import PolyMath
import tsv2ctf
import os
import argparse
import importlib

model_name = "pm.model"

def argument_by_name(func, name):
    found = [arg for arg in func.arguments if arg.name == name]
    if len(found) == 0:
        raise ValueError('no matching names in arguments')
    elif len(found) > 1:
        raise ValueError('multiple matching names in arguments')
    else:
        return found[0]
        
def create_mb_and_map(func, data_file, polymath, randomize=True, repeat=True):
    mb_source = C.io.MinibatchSource(
        C.io.CTFDeserializer(
            data_file,
            C.io.StreamDefs(
                context_g_words  = C.io.StreamDef('cgw', shape=polymath.wg_dim,     is_sparse=True),
                query_g_words    = C.io.StreamDef('qgw', shape=polymath.wg_dim,     is_sparse=True),
                context_ng_words = C.io.StreamDef('cnw', shape=polymath.wn_dim,     is_sparse=True),
                query_ng_words   = C.io.StreamDef('qnw', shape=polymath.wn_dim,     is_sparse=True),
                answer_begin     = C.io.StreamDef('ab',  shape=polymath.a_dim,      is_sparse=False),
                answer_end       = C.io.StreamDef('ae',  shape=polymath.a_dim,      is_sparse=False),
                context_chars    = C.io.StreamDef('cc',  shape=polymath.word_size,  is_sparse=False),
                query_chars      = C.io.StreamDef('qc',  shape=polymath.word_size,  is_sparse=False))),
        randomize=randomize,
        epoch_size=C.io.INFINITELY_REPEAT if repeat else C.io.FULL_DATA_SWEEP)

    input_map = {
        argument_by_name(func, 'cgw'): mb_source.streams.context_g_words,
        argument_by_name(func, 'qgw'): mb_source.streams.query_g_words,
        argument_by_name(func, 'cnw'): mb_source.streams.context_ng_words,
        argument_by_name(func, 'qnw'): mb_source.streams.query_ng_words,
        argument_by_name(func, 'cc' ): mb_source.streams.context_chars,
        argument_by_name(func, 'qc' ): mb_source.streams.query_chars,
        argument_by_name(func, 'ab' ): mb_source.streams.answer_begin,
        argument_by_name(func, 'ae' ): mb_source.streams.answer_end
    }
    return mb_source, input_map

def create_tsv_reader(func, tsv_file, polymath, seqs):
    with open(tsv_file, 'r', encoding='utf-8') as f:
        eof = False
        while not eof:
            batch={'cwids':[], 'qwids':[], 'baidx':[], 'eaidx':[], 'ccids':[], 'qcids':[]}
            
            while not eof and len(batch['cwids']) < seqs:
                line = f.readline()
                if not line:
                    eof = True
                    break
                    
                ctokens, qtokens, atokens, cwids, qwids,  baidx,   eaidx, ccids, qcids = tsv2ctf.tsv_iter(line, polymath.vocab, polymath.chars, False)

                batch['cwids'].append(cwids)
                batch['qwids'].append(qwids)
                batch['baidx'].append(baidx)
                batch['eaidx'].append(eaidx)
                batch['ccids'].append(ccids)
                batch['qcids'].append(qcids)
            
            if len(batch) > 0:
                context_g_words  = C.one_hot([[C.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in cwids] for cwids in batch['cwids']], polymath.wg_dim)
                context_ng_words = C.one_hot([[C.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in cwids] for cwids in batch['cwids']], polymath.wn_dim)
                query_g_words    = C.one_hot([[C.ONE_HOT_SKIP if i >= polymath.wg_dim else i for i in qwids] for qwids in batch['qwids']], polymath.wg_dim)
                query_ng_words   = C.one_hot([[C.ONE_HOT_SKIP if i < polymath.wg_dim else i - polymath.wg_dim for i in qwids] for qwids in batch['qwids']], polymath.wn_dim)
                context_chars = [[cc for cc in ccids+[-1]*max(0,polymath.word_size-len(ccids))] for ccids in batch['ccids']]
                query_chars   = [[qc for qc in qcids+[-1]*max(0,polymath.word_size-len(qcids))] for qcids in batch['qcids']]
                answer_begin = batch['baidx']
                answer_end = batch['eaidx']

                yield { argument_by_name(func, 'cgw'): context_g_words,
                        argument_by_name(func, 'qgw'): query_g_words,
                        argument_by_name(func, 'cnw'): context_ng_words,
                        argument_by_name(func, 'qnw'): query_ng_words,
                        argument_by_name(func, 'cc' ): context_chars,
                        argument_by_name(func, 'qc' ): query_chars,
                        argument_by_name(func, 'ab' ): answer_begin,
                        argument_by_name(func, 'ae' ): answer_end }
            

def train(data_path, model_path, log_file, config_file, restore=False, profiling=False):
    polymath = PolyMath(config_file)
    z, loss = polymath.model()
    training_config = importlib.import_module(config_file).training_config
    
    max_epochs = training_config['max_epochs']
    log_freq = training_config['log_freq']

    progress_writers = [C.logging.ProgressPrinter(
                            num_epochs = max_epochs,
                            freq = log_freq,
                            tag = 'Training',
                            log_to_file = log_file,
                            rank = C.Communicator.rank(),
                            gen_heartbeat = True)]

    C.set_default_use_mean_gradient_value(True)
    lr = C.learning_rate_schedule(training_config['lr'], unit=C.learners.UnitType.sample)
    learner = C.adadelta(z.parameters, lr)

    if C.Communicator.num_workers() > 1:
        learner = C.data_parallel_distributed_learner(learner, num_quantization_bits=32, distributed_after=0)

    trainer = C.Trainer(z, (loss, None), learner, progress_writers)

    if profiling:
        C.debugging.start_profiler(sync_gpu=True)

    train_data_file = os.path.join(data_path, training_config['train_data'])
    train_data_ext = os.path.splitext(train_data_file)[-1].lower()
    
    model_file = os.path.join(model_path, model_name)
    model = C.combine(list(z.outputs) + [loss.output])
    label_ab = argument_by_name(loss, 'ab')

    best_val_err = 100
    best_since = 0
    stop_after = training_config['stop_after']
    if train_data_ext == '.ctf':
        mb_source, input_map = create_mb_and_map(loss, train_data_file, polymath)

        minibatch_size = training_config['minibatch_size'] * C.Communicator.num_workers() # number of samples
        epoch_size = training_config['epoch_size']
        
        for epoch in range(max_epochs):
            num_seq = 0
            while True:
                data = mb_source.next_minibatch(minibatch_size, input_map=input_map, num_data_partitions=C.Communicator.num_workers(), partition_index=C.Communicator.rank())
                trainer.train_minibatch(data)
                num_seq += data[label_ab].num_sequences
                if num_seq >= epoch_size:
                    break

            trainer.summarize_training_progress()
            val_err = test_model(os.path.join(data_path, training_config['val_data']), model, polymath)
            if best_val_err > val_err:
                best_val_err = val_err
                best_since = 0
                if C.Communicator.rank() == 0:
                    model.save(model_file+'.best')
                    trainer.save_checkpoint(model_file+'.best')
                    trainer.save_checkpoint(model_file)
            else:
                best_since += 1
                if best_since > stop_after:
                    break # stop training when val_err did not go down for stop_after epochs

            if profiling:
                C.debugging.enable_profiler()
    else:
        if train_data_ext != '.tsv':
            raise Exception("Unsupported format")
        
        minibatch_seqs = training_config['minibatch_seqs'] # number of sequences

        for epoch in range(max_epochs):       # loop over epochs
            tsv_reader = create_tsv_reader(loss, train_data_file, polymath, minibatch_seqs)
            for data in tsv_reader:
                trainer.train_minibatch(data)                                   # update model with it

            trainer.summarize_training_progress()
            if profiling:
                C.debugging.enable_profiler()

        model.save(model_file)
    
    if profiling:
        C.debugging.stop_profiler()
        
def symbolic_best_span(begin, end):
    running_max_begin = C.layers.Recurrence(C.element_max, initial_state=-float("inf"))(begin)
    return C.layers.Fold(C.element_max, initial_state=C.constant(-1e+30))(running_max_begin + end)
        
def test_model(test_data, model, polymath):
    begin_logits = model.outputs[0]
    end_logits   = model.outputs[1]
    loss         = model.outputs[2]
    root = C.as_composite(loss.owner)
    mb_source, input_map = create_mb_and_map(root, test_data, polymath, randomize=False, repeat=False)
    begin_label = argument_by_name(root, 'ab')
    end_label   = argument_by_name(root, 'ae')

    begin_prediction = C.sequence.input(1, sequence_axis=begin_label.dynamic_axes[1], needs_gradient=True)
    end_prediction = C.sequence.input(1, sequence_axis=end_label.dynamic_axes[1], needs_gradient=True)
    
    best_span_score = symbolic_best_span(begin_prediction, end_prediction)
    predicted_span = C.layers.Recurrence(C.plus)(begin_prediction - C.sequence.past_value(end_prediction))
    true_span = C.layers.Recurrence(C.plus)(begin_label - C.sequence.past_value(end_label))
    common_span = C.element_min(predicted_span, true_span)
    begin_match = C.sequence.reduce_sum(C.element_min(begin_prediction, begin_label))
    end_match = C.sequence.reduce_sum(C.element_min(end_prediction, end_label))

    predicted_len = C.sequence.reduce_sum(predicted_span)
    true_len = C.sequence.reduce_sum(true_span)
    common_len = C.sequence.reduce_sum(common_span)
    f1 = 2*common_len/(predicted_len+true_len)
    exact_match = C.element_min(begin_match, end_match)
    precision = common_len/predicted_len
    recall = common_len/true_len
    overlap = C.greater(common_len, 0)
    s = lambda x: C.reduce_sum(x, axis=C.Axis.all_axes())
    stats = C.splice(s(f1), s(exact_match), s(precision), s(recall), s(overlap), s(begin_match), s(end_match))

    # Evaluation parameters
    minibatch_size = 8192
    num_sequences = 0

    stat_sum = 0
    loss_sum = 0
    
    C.debugging.start_profiler()
    C.debugging.enable_profiler()
    
    while True:
        data = mb_source.next_minibatch(minibatch_size, input_map=input_map)
        if not data or not (begin_label in data) or data[begin_label].num_sequences == 0:
            break
        out = model.eval(data, outputs=[begin_logits,end_logits,loss], as_numpy=False)
        testloss = out[loss]
        g = best_span_score.grad({begin_prediction:out[begin_logits], end_prediction:out[end_logits]}, wrt=[begin_prediction,end_prediction], as_numpy=False)
        other_input_map = {begin_prediction: g[begin_prediction], end_prediction: g[end_prediction], begin_label: data[begin_label], end_label: data[end_label]}
        stat_sum += stats.eval((other_input_map))
        loss_sum += np.sum(testloss.asarray())
        num_sequences += data[begin_label].num_sequences

    C.debugging.stop_profiler()

    stat_avg = stat_sum / num_sequences
    loss_avg = loss_sum / num_sequences

    print("Tested {} sequences, loss {:.4f}, F1 {:.4f}, EM {:.4f}, precision {:4f}, recall {:4f} hasOverlap {:4f}, start_match {:4f}, end_match {:4f}".format(
            num_sequences,
            loss_avg,
            stat_avg[0],
            stat_avg[1],
            stat_avg[2],
            stat_avg[3],
            stat_avg[4],
            stat_avg[5],
            stat_avg[6]))
            
    return loss_avg

def test(test_data, model_path, config_file):
    polymath = PolyMath(config_file)
    model = C.load_model(os.path.join(model_path, model_name))
    test_model(test_data, model, polymath)

if __name__=='__main__':
    # default Paths relative to current python file.
    abs_path   = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(abs_path, 'Models')
    data_path  = os.path.join(abs_path, '.')

    parser = argparse.ArgumentParser()
    parser.add_argument('-datadir', '--datadir', help='Data directory where the dataset is located', required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', default=False)
    parser.add_argument('-config', '--config', help='Config file', required=False, default='config')
    parser.add_argument('-r', '--restart', help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)', action='store_true')
    parser.add_argument('-test', '--test', help='Test data file', required=False, default=None)

    args = vars(parser.parse_args())

    if args['outputdir'] is not None:
        model_path = args['outputdir'] + "/models"
    if args['datadir'] is not None:
        data_path = args['datadir']

    test_data = args['test']
    
    if test_data == None:
        try:
            train(data_path, model_path, args['logdir'], args['config'],
                restore = not args['restart'],
                profiling = args['profile'])
        finally:
            C.distributed.Communicator.finalize()
    else:
        test(test_data, model_path, args['config'])
