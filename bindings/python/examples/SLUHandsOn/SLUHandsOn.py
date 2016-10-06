# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
import time
from cntk import DeviceDescriptor, Trainer, Axis, text_format_minibatch_source, StreamConfiguration
from cntk.learner import sgd
from cntk.ops import parameter, input_variable, placeholder_variable, times, cross_entropy_with_softmax, combine, classification_error
import itertools

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
from examples.common.nn import slice, sigmoid, log, tanh, past_value, future_value, print_training_progress, negate


#### temporary layers lib, to be moved out
from cntk.ops.functions import Function
from cntk.ops.variables import Variable

# "upgrade" a current Function to add additional operators and methods, as a temporary stopgap
# at end of each layer constructor, return _extend_Function(z, 'Type (for debugging)')
# use this until __call__ is implemented in Function()
# Also add the >> operator (forward function composition).
# Returns its arg to allow chaining.
def _extend_Function(f):
    class FunctionEx(f.__class__): 
        def __call__(self, *args):
            return _apply(self, _as_tuple(args))
        def __rshift__(self, other):
            return other(self)
        def _name(self):  # retrieve the debug name
            return _node_name(self)
    if hasattr(f, '__call__'):  # already extended: don't redo
        return f
    f.__class__ = FunctionEx
    print("def {}".format(_node_description(f)))
    return f

# name and extend; in this order, so that _extend_Function can print a meaningful log message
def _name_and_extend_Function(f, name):
    _name_node(f, name)
    _extend_Function(f)

# give a new name to a function, by wrapping it
def _wrap_rename_Function(f, name):
     f = combine([f]) ; _name_and_extend_Function(f, name)  # 'combine' to create a separate identity so we can reassign the debug name
     return f

# upgrade Trainer class, add new method
def _extend_Trainer(trainer):
    class TrainerEx(trainer.__class__):
        # new method get_next_minibatch()
        # TODO: make this a true static method so we can say Trainer.get_next_minibatch()
        # TODO: is the "_next" really necessary? Trainer.get_minibatch() seems sufficient
        @staticmethod
        def get_next_minibatch(source, minibatch_size, input_map):
            mb = reader.get_next_minibatch(minibatch_size)
            if len(mb) == 0:  # TODO: return None instead?
                return None
            else:
                return { key : mb[value].m_data for (key, value) in input_map.items() }
    if hasattr(trainer, 'get_next_minibatch'):  # already extended: don't redo
        return trainer
    trainer.__class__ = TrainerEx
    return trainer

# helper to name nodes for printf debugging
_auto_node_names = dict()
_auto_name_count = dict()
def _name_node(n, name):
    if not n in _auto_node_names:     # only name a node once
        # strip _.*
        #name = name.split('[')[0]
        if not name in _auto_name_count: # count each type separately
            _auto_name_count[name] = 1
        else:
            _auto_name_count[name] += 1
        #name = name + "[{}]".format(_auto_name_count[name])
        name = name + ".{}".format(_auto_name_count[name])
        try:
            name = name + n.uid()
        except:
            pass
        _auto_node_names[n] = name
    return n

# this gives a name to anything not yet named
def _node_name(n):
    global _auto_node_names, _auto_name_count
    if n in _auto_node_names:
        return _auto_node_names[n]
    try:
        name = n.what()
    except:
        name = n.name()
    # internal node names (not explicitly named)
    if name == '':
        if hasattr(n, 'is_placeholder') and n.is_placeholder:
            name = '_'
        else:
            name = '_f'
    _name_node(n, name)
    return _node_name(n)

# -> node name (names of function args if any)
def _node_description(n):
    desc = _node_name(n)
    if hasattr(n, 'inputs'):
        inputs = n.inputs()
        #desc = "{} [{}]".format(desc, ", ".join([_node_name(p) for p in inputs]))
        func_params = [input for input in inputs if input.is_parameter()]
        func_args   = [input for input in inputs if input.is_placeholder()]
        if func_params:
            desc = "{} {{{}}}".format(desc, ", ".join([_node_name(p) for p in func_params]))
        desc = "{} <{}>".format(desc, ", ".join([_node_name(func_arg) for func_arg in func_args]))
    return desc

def _print_node(n):
    print (_node_description(n))

#def dump_graph(f):
#    visited = set()
#    def r_dump_graph(f, indent):
#        if f in visited:  # don't double-print
#            return
#        visited.add(f)
#        # print a node
#        inputs = f.root_function.inputs()
#        s = "{} ( ".format(f.name())
#        for c in inputs:
#            s += _node_name(c) + " "
#        s += ")"
#        print(s)
#        # print its children
#        for c in inputs:
#            r_dump_graph(c, indent+2)
#    r_dump_graph (f, 0)

# monkey-patching some missing stuff
def __matmul__(a,b):  # TODO: define @ once we have Python 3.5
    return times(a,b)
#Function.__matmul__ = __matmul__  # should work in Python 3.5  --Function is not defined?

# helper to convert a dictionary into a Python class, so that the dict looks like an immutable record
class _ClassFromDict(dict):
    def __init__(self, dict):
        super(_ClassFromDict, self).__init__(dict)
        for key in dict:
            self[key] = dict[key]
    def __getattr__(self, k):
        return self.get(k)

# easier construction of records
# e.g. r = Record(x = 13, y = 42) ; x = r.x
def Record(**kwargs):
    return _ClassFromDict(kwargs)

# type-cast a shape given as a scalar into a tuple
def _as_tuple(x):
    return x if (isinstance(x,tuple)) else (x,)
def _Infer(shape, axis):
    return Record(shape=_as_tuple(shape), axis=axis, with_shape = lambda new_shape: _Infer(new_shape, axis))

# returns True if x is a Variable object
#def is_variable(x):
#    return hasattr(x, 'is_placeholder')
#    #return isinstance(x, cntk.cntk_py.Variable)

# TODO: should no longer be needed
#def _get_placeholders(f):
#    if is_variable(f):
#        return [f];
#    try:
#        return f.placeholders()
#    except AttributeError: # Workaround: in Python, a Placeholder cannot auto-cast to an identity function
#        return [f] # BUGBUG: only correct if not bound yet. But this problem shall go away anyway.

def _apply(f, args):
    import operator   # add()
    import functools  # reduce()
    from cntk.cntk_py import ParameterCloningMethod_Share
    # flatten args to a list. Note it may be a a tuple or even a nested tree of tuples, e.g. LSTM (x, (h, c))
    def flatten_tuple(args):
        if not isinstance(args, tuple): # not a tuple: singleton; create a singleton tuple
            return (args,)
        return functools.reduce(operator.add, [(flatten_tuple(item)) for item in args])
    args = list(flatten_tuple(args))
    def _output_of(arg):  # helper to get the output of an arg; use arg itself if no output() method (that'd be a Variable)
        try:
            return arg.output()
        except AttributeError:
            return arg  # Variables have no output()
    args = [_output_of(arg) for arg in args]
    # for printf debugging
    try:
        placeholders = f.placeholders()  # f parameters to fill in
        #print (len(args))
        #print (len(placeholders))
        if len(args) != len(placeholders):
            raise TypeError("_apply ({}): number of arguments {} must match number of placeholders {}".format(_node_description(f), len(args), len(placeholders)))
        fn = _node_name(f)
        fw = _node_description(f)
        aw = ", ".join([_node_name(f) for f in list(args)])
        f = f.clone(ParameterCloningMethod_Share)
        f.replace_placeholders(dict(zip(f.placeholders(), args)))
        #c = f.clone(dict(zip(placeholders, args)))
        _name_and_extend_Function(f, fn)
        print("{} = {} ({})".format(_node_description(f), fw, aw))
        return f
        #return f.clone(dict(zip(placeholders, args)))
        # BUGBUG: ^^ fails with TypeError: clone() takes 1 positional argument but 2 were given
    except AttributeError: # Workaround: a Variable object can be passed here, and it can be a placeholder itself
        #if f.is_placeholder():
        #    return args[0]  # identity function
        #else:
            raise

# some mappings to BS format
def Parameter(shape, learning_rate_multiplier=1.0, init=None, init_value_scale=1, init_value=None, init_filter_rank=0, init_output_rank=1, init_from_file_path=None, init_on_cpu_only=True, random_seed=-1):
    return _name_node(parameter(shape), 'parameter')   # these are factory methods for things with state
def Input(*args, **kwargs):
    return _name_node(input_variable(*args, **kwargs), 'input')

def Placeholder(_inf, name='placeholder'):
    # BUGBUG: does not take a name parameter
    # BUGBUG: combine() does not work either
    p = placeholder_variable(shape=_as_tuple(_inf.shape), dynamic_axes=_inf.axis)
    _name_node(p, name)
    print("new " + _node_description(p))
    return p

# Sequential -- composite that applies a sequence of functions onto an input
# Sequential ([F, G, H]) === F >> G >> H
def Sequential(arrayOfFunctions, _inf):
    #import functools  # reduce()
    #return functools.reduce(lambda g, f: _name_and_extend_Function(f, 'Sequential element') >> g, arrayOfFunctions)
    #r = Placeholder(_inf=_inf)
    r = layers.Identity(_inf=_inf)
    for f in arrayOfFunctions:
        #_print_node(r)
        r = r >> f
    apply_x = _wrap_rename_Function(r, 'Sequential')
    return apply_x;

class layers:
    # need to define everything indented by 4

    #_INFERRED = 0   # TODO: use the predefined name for this

    # Linear -- create a fully-connected linear projection layer
    # Note: shape may describe a tensor as well.
    # TODO: change to new random-init descriptor
    @staticmethod
    def Linear(shape, _inf, bias=True, init='glorot_uniform', init_value_scale=1, input_rank=None, map_rank=None):
        out_shape = _as_tuple(shape)
        W = Parameter(_inf.shape + out_shape, init=init, init_value_scale=init_value_scale)
        b = Parameter(             out_shape, init='zero') if bias else None
        x = Placeholder(_inf=_inf, name='linear_arg')
        apply_x = __matmul__(x, W) + b if bias else \
                  __matmul__(x, W)
        _name_and_extend_Function(apply_x, 'Linear')
        return apply_x
        # TODO: how to break after the else?

    # Embedding -- create a linear embedding layer
    @staticmethod
    def Embedding(shape, _inf, init='glorot_uniform', init_value_scale=1, embedding_path=None, transpose=False):
        shape = _as_tuple(shape)
        full_shape = (shape + _inf.shape) if transpose else (_inf.shape + shape)
        if embedding_path is None:
            # TODO: how to pass all optional args automatically in one go?
            f = layers.Linear(shape, _inf=_inf, init=init, init_value_scale=init_value_scale)
            _wrap_rename_Function(f, 'Embedding')
            return f
        else:
            E = Parameter(full_shape, initFromFilePath=embeddingPath, learningRateMultiplier=0)  # fixed from file
        _ = Placeholder(_inf=_inf, name='embedding_arg')
        apply_x = __matmul__(E, _) if transposed else \
                __matmul__(_, E)     # x is expected to be sparse one-hot
        _name_and_extend_Function(apply_x, 'Embedding')
        return apply_x

    @staticmethod
    def Stabilizer(_inf, steepness=4):
        # sharpened Softplus: 1/steepness ln(1+e^{steepness*beta})
        # this behaves linear for weights around 1, yet guarantees positiveness

        # parameters
        param = Parameter((1), init_value=0.99537863)  # 1/steepness*ln (e^steepness-1) for steepness==4

        # application
        x = Placeholder(_inf=_inf, name='stabilizer_arg')
        beta = log (1 + exp (steepness * param)) / steepness
        apply_x = beta * x
        _name_and_extend_Function(apply_x, 'Stabilizer')
        return apply_x

    @staticmethod
    def Identity(_inf):
        x = Placeholder(_inf=_inf, name='identity_arg')
        #apply_x = combine([x])  # BUGBUG: not working
        apply_x = x + 0
        _name_and_extend_Function(apply_x, 'Identity')
        return apply_x

    # TODO: Fornow, shape and cell_shape can only be rank-1 vectors
    @staticmethod
    def LSTMBlock(shape, _inf, cell_shape=None, use_peepholes=False, init='glorot_uniform', init_value_scale=1, enable_self_stabilization=False): # (x, (h, c))
        has_projection = cell_shape is not None
        has_aux = False

        shape = _as_tuple(shape)

        cell_shape = _as_tuple(cell_shape) if cell_shape is not None else shape

        stack_axis = 0  # TODO: make this -1, i.e. the fastest-changing one, to match BS
        # determine stacking dimensions
        cell_shape_list = list(cell_shape)
        stacked_dim = cell_shape_list[0]
        cell_shape_list[stack_axis] = stacked_dim*4
        cell_shape_stacked = tuple(cell_shape_list)  # patched dims with stack_axis duplicated 4 times

        # parameters
        B  = Parameter(             cell_shape_stacked, init_value=0)       # a bias
        W  = Parameter(_inf.shape + cell_shape_stacked, init=init, init_value_scale=init_value_scale)                             # input
        A  = Parameter(_inf.shape + cell_shape_stacked, init=init, init_value_scale=init_value_scale) if has_aux else None        # aux input (optional)
        H  = Parameter(shape      + cell_shape_stacked, init=init, init_value_scale=init_value_scale)                             # hidden-to-hidden
        Ci = Parameter(             cell_shape,         init=init, init_value_scale=init_value_scale) if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}
        Cf = Parameter(             cell_shape,         init=init, init_value_scale=init_value_scale) if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}
        Co = Parameter(             cell_shape,         init=init, init_value_scale=init_value_scale) if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}

        Wmr = ParameterTensor (cell_shape + shape, init=init, init_value_scale=init_value_scale) if has_projection else None  # final projection

        Sdh = layers.Stabilizer(_inf=_inf.with_shape(     shape)) if enable_self_stabilization else layers.Identity(_inf=_inf.with_shape(     shape))
        Sdc = layers.Stabilizer(_inf=_inf.with_shape(cell_shape)) if enable_self_stabilization else layers.Identity(_inf=_inf.with_shape(cell_shape))
        Sct = layers.Stabilizer(_inf=_inf.with_shape(cell_shape)) if enable_self_stabilization else layers.Identity(_inf=_inf.with_shape(cell_shape))
        Sht = layers.Stabilizer(_inf=_inf.with_shape(     shape)) if enable_self_stabilization else layers.Identity(_inf=_inf.with_shape(     shape))

        def create_hc_placeholder():
            return (Placeholder(_inf=_inf.with_shape(shape), name='hPh'), Placeholder(_inf=_inf.with_shape(cell_shape), name='cPh')) # (h, c)

        # parameters to model function
        x = Placeholder(_inf=_inf, name='lstm_block_arg')
        prev_state = create_hc_placeholder()

        # formula of model function
        dh, dc = prev_state

        dhs = Sdh(dh)  # previous values, stabilized
        dcs = Sdc(dc)
        # note: input does not get a stabilizer here, user is meant to do that outside

        # projected contribution from input(s), hidden, and bias
        proj4 = B + times(x, W) + times(dhs, H) + times(aux, A) if has_aux else \
                B + times(x, W) + times(dhs, H)

        it_proj  = slice (proj4, stack_axis, 0*stacked_dim, 1*stacked_dim)  # split along stack_axis
        bit_proj = slice (proj4, stack_axis, 1*stacked_dim, 2*stacked_dim)
        ft_proj  = slice (proj4, stack_axis, 2*stacked_dim, 3*stacked_dim)
        ot_proj  = slice (proj4, stack_axis, 3*stacked_dim, 4*stacked_dim)
        #it_proj  = proj4  #, stack_axis, 0*stacked_dim, 1*stacked_dim)  # split along stack_axis
        #bit_proj = proj4  #, stack_axis, 1*stacked_dim, 2*stacked_dim)
        #ft_proj  = proj4  #, stack_axis, 2*stacked_dim, 3*stacked_dim)
        #ot_proj  = proj4  #, stack_axis, 3*stacked_dim, 4*stacked_dim)
        # BUGBUG: replace_placeholders fails with TypeError: slice() got multiple values for argument 'axis'

        # add peephole connection if requested
        def peep(x, c, C):
            return x + C * c if use_peepholes else x

        it = sigmoid (peep (it_proj, dcs, Ci))        # input gate(t)
        bit = it * tanh (bit_proj)                    # applied to tanh of input network

        ft = sigmoid (peep (ft_proj, dcs, Cf))        # forget-me-not gate(t)
        bft = ft * dc                                 # applied to cell(t-1)

        ct = bft + bit                                # c(t) is sum of both

        ot = sigmoid (peep (ot_proj, Sct(ct), Co))    # output gate(t)
        ht = ot * tanh (ct)                           # applied to tanh(cell(t))

        c = ct          # cell value
        h = times(Sht(ht), Wmr) if has_projection else \
            ht

        _name_node(h, 'h')
        _print_node(h)  # this looks right
        _name_node(c, 'c')

        # return to caller a helper function to create placeholders for recurrence
        apply_x_h_c = combine ([h, c])
        apply_x_h_c.create_placeholder = create_hc_placeholder
        _name_and_extend_Function(apply_x_h_c, 'LSTMBlock')
        return apply_x_h_c

    @staticmethod
    def Recurrence(over=None, _inf=None, go_backwards=False):
        # helper to compute previous value
        # can take a single Variable/Function or a tuple
        def previous_hook(state):
            if hasattr(state, 'outputs'):
               outputs = state.outputs()
               if len(outputs) > 1:  # if multiple then apply to eaAntoher ch element
                   return tuple([previous_hook(s) for s in outputs])
            # not a tuple: must be a 'scalar', i.e. a single element
            return past_value(state) if not go_backwards else \
                   future_value(state)
        x = Placeholder(_inf=_inf, name='recurrence_arg')
        prev_state_forward = over.create_placeholder() # create a placeholder or a tuple of placeholders
        f_x_h_c = over(x, prev_state_forward) # apply the recurrent over
        # this returns a Function (x, (h_prev, c_prev)) -> (h, c)
        h = f_x_h_c.outputs()[0]  # 'h' is a Variable (the output of a Function that computed it)
        _print_node(h)
        _print_node(combine([h.owner()]))#)
        prev_state = previous_hook(f_x_h_c)  # delay (h, c)
        repl_list = {value_forward: value.output() for (value_forward, value) in list(zip(list(prev_state_forward), list(prev_state)))}
        #repl_list = {value_forward: value for (value_forward, value) in list(zip(list(prev_state_forward), list(prev_state)))}
        f_x_h_c.replace_placeholders(repl_list)  # binds _h_c := prev_state
        apply_x = combine([h.owner()])     # the Function that yielded 'h', so we get to know its inputs
        # apply_x is a Function x -> h
        _name_and_extend_Function(apply_x, 'Recurrence')
        _print_node(apply_x)
        return apply_x

#### User code begins here

cntk_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../../.."  # CNTK root folder
data_dir = cntk_dir + "/Tutorials/SLUHandsOn"

model_dir = "./Models"

vocab_size = 943 ; num_labels = 129 ; num_intents = 26    # number of words in vocab, slot labels, and intent labels

def TextFormatMinibatchSource(path, stream_defs, epoch_size):
    # convert stream_defs into StreamConfiguration format
    stream_configs = [ StreamConfiguration(key, dim=value.dim, is_sparse=value.is_sparse, stream_alias=value.stream_alias) for (key, value) in stream_defs.items() ]
    source = text_format_minibatch_source(path, stream_configs, epoch_size)
    # attach a dictionary of the streams
    source.streams = _ClassFromDict({ name : source.stream_info(name) for name in stream_defs.keys() })
    return source
StreamDef = Record

def Reader(path):
    return TextFormatMinibatchSource(path,
               Record(
                   query         = StreamDef(dim=vocab_size, is_sparse=True, stream_alias='S0'),
                   intent_unused = StreamDef(dim=10000,      is_sparse=True, stream_alias='S1'),  # BUGBUG: unused, and should infer dim
                   slot_labels   = StreamDef(dim=num_labels, is_sparse=True, stream_alias='S2')
               ),
               epoch_size=36000)  # TODO: move this up
    # what's that 36000 at the end; is that the epoch size?
    # stream_alias -> alias
    # if the dimension is 'dim', should it be called that in the other places as well, instead of 'shape'? Or change to 'shape'?

input_dim  = vocab_size
emb_dim    = 150
hidden_dim = 300
label_dim  = num_labels

def Model(_inf):
    return Sequential([
        layers.Embedding(shape=emb_dim, _inf=_inf),
        layers.Recurrence(over=layers.LSTMBlock(shape=hidden_dim, _inf=_inf.with_shape(emb_dim)), _inf=_inf.with_shape(emb_dim), go_backwards=False),
        layers.Linear(shape=label_dim, _inf=_inf.with_shape(hidden_dim))
    ], _inf=_inf)

def train(reader, model):
    # Input variables denoting the features and label data
    query       = Input(input_dim,  is_sparse=False)  # TODO: make sparse once it works
    slot_labels = Input(num_labels, is_sparse=True)

    # apply model to input
    z = model(query)

    # loss and metric
    ce = cross_entropy_with_softmax(z, slot_labels)
    pe = classification_error      (z, slot_labels)

    # training config
    lr = 0.003  # TODO: [0.003]*2 + [0.0015]*12 + [0.0003]
    #gradUpdateType = "fsAdaGrad"
    #gradientClippingWithTruncation = True ; clippingThresholdPerSample = 15.0
    #first_mbs_to_show_result = 10
    minibatch_size = 70
    num_mbs_to_show_result = 10

    # trainer object
    trainer = Trainer(z, ce, pe, [sgd(z.parameters(), lr)])
    _extend_Trainer(trainer)

    # define mapping from reader streams to network inputs
    # TODO: how to do epochs??
    input_map = {
        query       : reader.streams.query,
        slot_labels : reader.streams.slot_labels
    }

    # process minibatches and perform model training
    for i in itertools.count():
        data = trainer.get_next_minibatch(reader, minibatch_size, input_map)
        if data is None:
            break
        trainer.train_minibatch(data)
        print_training_progress(trainer, i, num_mbs_to_show_result)

def set_gpu(gpu_id):
    # Specify the target device to be used for computing
    target_device = DeviceDescriptor.gpu_device(gpu_id)
    DeviceDescriptor.set_default_device(target_device)

if __name__=='__main__':
    set_gpu(0)
    reader = Reader(data_dir + "/atis.train.ctf")
    model = Model(_inf=_Infer(shape=input_dim, axis=[Axis.default_batch_axis(), Axis.default_dynamic_axis()]))
    # BUGBUG: Currently this fails with a mismatch error if axes ^^ are given in opposite order
    train(reader, model)
