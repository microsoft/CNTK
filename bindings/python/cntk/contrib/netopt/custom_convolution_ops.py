# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
import numpy as np
import cntk as C
from cntk.ops.functions import UserFunction

# custom sign function using the straight through estimator as gradient. This is implemented without a numpy intermediate
# representation, giving a much faster run time.
class SignWithEstimation(UserFunction):
    # initialize by creating new userfunction and assigning function and grad inputs and functions
    def __init__(self, arg, name='SignWithEstimation'):
        super(SignWithEstimation, self).__init__([arg], as_numpy=False, name=name)
        # create an input variable and a function object for the forward propagated function
        self.action, self.actionArg = self.signFunc(arg)
        # create a binary input gradient function, two input variables and the function object. gradroot is the incoming
        # gradient from stages down the pipeline and gradarg is the argument we need to do our new gradient. In our
        # case, we need the inputs to the forward function.
        self.grad, self.gradArg, self.gradRoot = self.gradFunc(arg)

    # define the forward propagation function y = sign(x)
    def signFunc(self, arg):
        # create an input variable that matches the dimension of the input argument        
        signIn = C.input(shape=arg.shape, dynamic_axes=arg.dynamic_axes)
        # create the first stage of the sign function, check if input is greater than zero
        actionfunc = C.greater(signIn, 0)
        # return the second stage of the sign function, replace any 0s with -1s
        return C.element_select(actionfunc, actionfunc, -1), signIn

    # define the backward propagation function, delta_out = delta_in * (|x| <= 1) ? 1 : 0.
    def gradFunc(self, arg):
        # create an input variable corresponding the inputs of the forward prop function
        gradIn = C.input(shape=arg.shape, dynamic_axes=arg.dynamic_axes)
        # create an input variable for the gradient passed from the next stage
        gradRoot = C.input(shape=arg.shape, dynamic_axes=arg.dynamic_axes)
        # first step is to take absolute value of input arg
        signGrad = C.abs(gradIn)
        # then compare its magnitude to 1
        signGrad = C.less_equal(signGrad, 1)
        # finish by multiplying this result with the input gradient
        return C.element_times(gradRoot, signGrad), gradIn, gradRoot

    # define what should happen when a customsign function object is forwarded
    def forward(self, argument, device, outputs_to_retain):
        # perform forward on the action function object. To do this, we have to map argument to the actions input,
        # actionArg, we specify that the outputs should be stored in action.outputs and set as_numpy to false to get
        # things running faster
        _, output_values = self.action.forward({self.actionArg: argument}, self.action.outputs, device=device,
                                               as_numpy=False)
        # first return argument is what to store in state, in our case, we need to keep the inputs to forward to compute
        # the straight through estimator gradient. Second output is simply selecting the proper output from output_values.
        return argument.deep_clone(), output_values[self.action.output]

    # define what should happend when our function has backward performed on it
    def backward(self, state, root_gradients):
        # first extract the value passed by the state, in this case the argument to forward.
        val = state
        # now perform a forward on our gradient function to compute the proper outputs. Map val to gradArg and the
        # root gradient to gradRoot to properly set up the binary gradient inputs.
        _, output_values = self.grad.forward({self.gradArg: val, self.gradRoot: root_gradients},
                                             self.grad.outputs, device=state.device(), as_numpy=False)
        # return the proper output
        return C.output_values[self.grad.output]

    # define a function that returns an output_variable with the shape of this function
    def infer_outputs(self):
        return [C.output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]


# A different implementation of sign with the straight through estimator for backprop. Has identical outputs but uses numpy instead
# of CNTK intrinsics. This makes it much easier to write but slower for training since data has to be copied between CPU and GPU.
class pySign(UserFunction):
    def __init__ (self, arg, name='pySign'):
        super(pySign, self).__init__([arg], name=name)

    def forward(self, argument, device=None, outputs_to_retain=None):
        sign = np.sign(argument)
        np.place(sign, sign==0, -1)
        return argument, sign

    def backward(self, state, root_gradients):
        input = state
        grad = np.abs(input)
        grad = np.less_equal(grad, 1)
        return grad*root_gradients

    def infer_outputs(self):
        return [C.output_variable(self.inputs[0].shape, self.inputs[0].dtype,
                                self.inputs[0].dynamic_axes)]

# Similar to CustomSign, Multibit binarizes an input using straight through estimator gradients. However, Multibit also supports a 
# bit_map argument that specifies how many bits to binarize to. Bit_map is a tensor the shape of the input that has a bit value for
# each input value. Although the bit_map will be uniform in most cases, it does support varying bit widths. The kernel variant
# of Multibit computes a scaler for each bit for each kernel. Should only be used on weight values. For activations, use the regular
# Multibit version.
class MultibitKernel(UserFunction):
    # initialize by creating new userfunction and assigning function and grad inputs and functions
    def __init__(self, arg1, arg2, name='MultibitKernel'):
        super(MultibitKernel, self).__init__([arg1], as_numpy=False, name=name)
        # save the input bit map for later
        self.bit_map = arg2
        # create an input variable and a function object for the forward propagated function
        self.action, self.actionArg = self.multiFunc(arg1)
        # create a binary input gradient function, two input variables and the function object. gradroot is the incoming
        # gradient from stages down the pipeline and gradarg is the argument we need to do our new gradient. In our
        # case, we need the inputs to the forward function.
        self.grad, self.gradArg, self.gradRoot = self.gradFunc(arg1)

    # define the forward propagation function y = sign(x)
    def multiFunc(self, arg1):
        # load or create the inputs we need
        multiIn = C.input(shape=arg1.shape, dynamic_axes = arg1.dynamic_axes)
        bit_map = C.constant(self.bit_map)
        max_bits = self.bit_map.max()
        shape = multiIn.shape
        reformed = C.reshape(multiIn, (-1,))
        # lets compute the means we need
        # carry over represents the remaining value that needs to binarized. For a single bit, this is just the input. For more bits,
        # it is the difference between the previous bits approximation and the true value.
        carry_over = multiIn
        approx = C.element_times(multiIn, 0)
        # iterate through the maximum number of bits specified by the bit maps, basically compute each level of binarization
        for i in range(max_bits):
            # determine which values of the input should be binarized to i bits or more            
            hot_vals = (np.greater(bit_map, i)).as_type(float)
            # select only the values which we need to binarize
            valid_vals = C.element_select(hot_vals, carry_over, 0)
            # compute mean on a per kernel basis, reshaping is done to allow for sum reduction along only axis 0 (the kernels)
            mean = C.element_divide(C.reduce_sum(C.reshape(C.abs(valid_vals), (valid_vals.shape[0], -1)), axis=1), C.reduce_sum(C.reshape(hot_vals, (hot_vals.shape[0], -1)), axis=1))
            # reshape the mean to match the dimensionality of the input
            mean = C.reshape(mean, (mean.shape[0], mean.shape[1], 1, 1))
            # binarize the carry over
            bits = C.greater(carry_over, 0)
            bits = C.element_select(bits, bits, -1)
            bits = C.element_select(hot_vals, bits, 0)
            # add in the equivalent binary representation to the approximation
            approx = C.plus(approx, C.element_times(mean, bits))
            # compute the new carry over
            carry_over = C.plus(C.element_times(C.element_times(-1, bits), mean), carry_over)
            
        return approx, multiIn

    # define the backward propagation function, delta_out = delta_in * (|x| <= 1) ? 1 : 0.
    def gradFunc(self, arg):
        # create an input variable corresponding the inputs of the forward prop function
        gradIn = C.input(shape=arg.shape, dynamic_axes=arg.dynamic_axes)
        # create an input variable for the gradient passed from the next stage
        gradRoot = C.input(shape=arg.shape, dynamic_axes=arg.dynamic_axes)
        signGrad = C.abs(gradIn)
        # new idea, bound of clipping should be a function of the bit map since higher bits can represent higher numbers
        bit_map = C.constant(self.bit_map)
        signGrad = C.less_equal(signGrad, bit_map)
        outGrad = signGrad

        outGrad = C.element_times(gradRoot, outGrad)
    
        return outGrad, gradIn, gradRoot

    # define what should happen when a customsign function object is forwarded
    def forward(self, argument, device, outputs_to_retain):
        # perform forward on the action function object. To do this, we have to map argument to the actions input,
        # actionArg, we specify that the outputs should be stored in action.outputs and set as_numpy to false to get
        # things running faster
        _, output_values = self.action.forward({self.actionArg: argument}, self.action.outputs, device=device,
                                               as_numpy=False)
        # first return argument is what to store in state, in our case, we need to keep the inputs to forward to compute
        # the straight through estimator gradient. Second output is simply selecting the proper output from output_values.
        return argument.deep_clone(), output_values[self.action.output]

    # define what should happend when our function has backward performed on it
    def backward(self, state, root_gradients):
        # first extract the value passed by the state, in this case the argument to forward.
        val = state
        # now perform a forward on our gradient function to compute the proper outputs. Map val to gradArg and the # root gradient to gradRoot to properly set up the binary gradient inputs.
        _, output_values = self.grad.forward({self.gradArg: val, self.gradRoot: root_gradients},
                                             self.grad.outputs, device=state.device(), as_numpy=False)
        # return the proper output
        return output_values[self.grad.output]

    # define a function that returns an output_variable with the shape of this function
    def infer_outputs(self):
        return [C.output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]



# Similar to Multibit kernel but computes a single input-wide scaler per bit
class Multibit(UserFunction):
    # initialize by creating new userfunction and assigning function and grad inputs and functions
    def __init__(self, arg1, arg2, name='Multibit'):
        super(Multibit, self).__init__([arg1], as_numpy=False, name=name)
        self.bit_map = arg2

    def multiFunc(self, arg1):
        multiIn = C.input(shape=arg1.shape, dynamic_axes = arg1.dynamic_axes)
        bit_map = C.constant(self.bit_map)
        max_bits = self.bit_map.max()
        shape = multiIn.shape
        reformed = C.reshape(multiIn, (-1,))
        carry_over = multiIn
        approx = C.element_times(multiIn, 0)
        for i in range(max_bits):
            hot_vals = C.greater(bit_map, i)
            valid_vals = C.element_select(hot_vals, carry_over, 0)
            mean = C.element_divide(C.reduce_sum(C.abs(valid_vals)), C.reduce_sum(hot_vals))
            bits = C.greater(carry_over, 0)
            bits = C.element_select(bits, bits, -1)
            bits = C.element_select(hot_vals, bits, 0)
            approx = C.plus(approx, C.element_times(mean, bits))
            carry_over = C.plus(C.element_times(C.element_times(-1, bits), mean), carry_over)
            
        return approx, multiIn

    # define the backward propagation function, delta_out = delta_in * (|x| <= 1) ? 1 : 0.
    def gradFunc(self, arg):
        # create an input variable corresponding the inputs of the forward prop function
        gradIn = C.input(shape=arg.shape, dynamic_axes=arg.dynamic_axes)
        # create an input variable for the gradient passed from the next stage
        gradRoot = C.input(shape=arg.shape, dynamic_axes=arg.dynamic_axes)
  
        signGrad = C.abs(gradIn)
        # new idea, bound of clipping should be a function of the bit map since higher bits can represent higher numbers
        bit_map = C.constant(self.bit_map)
        signGrad = C.less_equal(signGrad, bit_map)
        outGrad = signGrad

        outGrad = C.element_times(gradRoot, outGrad)
    
        return outGrad, gradIn, gradRoot

    # define what should happen when a customsign function object is forwarded
    def forward(self, argument, device, outputs_to_retain):
        # perform forward on the action function object. To do this, we have to map argument to the actions input,
        # actionArg, we specify that the outputs should be stored in action.outputs and set as_numpy to false to get
        # things running faster
        _, output_values = self.action.forward({self.actionArg: argument}, self.action.outputs, device=device,
                                               as_numpy=False)
        # first return argument is what to store in state, in our case, we need to keep the inputs to forward to compute
        # the straight through estimator gradient. Second output is simply selecting the proper output from output_values.
        return argument.deep_clone(), output_values[self.action.output]

    # define what should happend when our function has backward performed on it
    def backward(self, state, root_gradients):
        # first extract the value passed by the state, in this case the argument to forward.
        val = state
        # now perform a forward on our gradient function to compute the proper outputs. Map val to gradArg and the # root gradient to gradRoot to properly set up the binary gradient inputs.
        _, output_values = self.grad.forward({self.gradArg: val, self.gradRoot: root_gradients},
                                             self.grad.outputs, device=state.device(), as_numpy=False)
        # return the proper output
        return output_values[self.grad.output]

    # define a function that returns an output_variable with the shape of this function
    def infer_outputs(self):
        output_vars = [C.output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

        self.action, self.actionArg = self.multiFunc(self.inputs[0])
        self.grad, self.gradArg, self.gradRoot = self.gradFunc(self.inputs[0])

        return output_vars

    def serialize(self):
        return {'bit_map': np.asarray(self.bit_map, dtype=np.float32)}

    @staticmethod
    def deserialize(inputs, name, state):
        return Multibit(inputs[0], np.asarray(state['bit_map'], dtype=np.int32), name)

    def clone(self, cloned_inputs):
        cloned_inputs[0].__class__ = C.Variable
        return Multibit(cloned_inputs[0], self.bit_map, self.name)

# these are the face of the custom functions, they simply instantiate a custom function by calling user_function
def CustomSign(input):
    return C.user_function(SignWithEstimation(input))

def CustomPySign(input):
    return C.user_function(pySign(input))

def CustomMultibit(input, bit_map, mean_bits=None):
    if (mean_bits):
        bit_map = np.asarray(np.maximum(np.round(np.random.normal(mean_bits, 1, input.shape)), 1), dtype=np.int32)
        print("Mean Bits: ",np.mean(bit_map))
    else:
        if (type(bit_map) == int):
            length = C.reshape(input, (-1))
            bit_map = [bit_map]*length.shape[0]
            bit_map = np.asarray(bit_map)
            bit_map = bit_map.reshape(input.shape)
        else:
            bit_map = np.asarray(bit_map)
    assert (bit_map.shape == input.shape)
    return C.user_function(Multibit(input, bit_map))

def CustomMultibitKernel(input, bit_map, mean_bits=None):
    if (mean_bits):
        bit_map = np.asarray(np.maximum(np.round(np.random.normal(mean_bits, 1, input.shape)), 1), dtype=np.int32)
        print("Mean Bits: ",np.mean(bit_map))
    else:
        if (type(bit_map) == int):
            length = C.reshape(input, (-1))
            bit_map = [bit_map]*length.shape[0]
            bit_map = np.asarray(bit_map)
            bit_map = bit_map.reshape(input.shape)
        else:
            bit_map = np.asarray(bit_map)
    assert (bit_map.shape == input.shape)
    return C.user_function(MultibitKernel(input, bit_map))
