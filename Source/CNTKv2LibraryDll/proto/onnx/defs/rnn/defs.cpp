#include "proto/onnx/core/op.h"

namespace ONNXIR {

    std::function<void(OperatorSchemaSetter&)> RNNDocGenerator() {
        return [=](OperatorSchemaSetter& schema) {
            schema.Input("X",
                "The input sequences packed (and potentially padded) into one 3-D "
                "tensor with the shape of `[seq_length, batch_size, input_size]`.", "T");
            schema.Input("sequence_lens",
                "Optional tensor specifying lengths of the sequences in a batch. "
                "If not specified - assumed all sequences in the batch to have "
                "length `seq_length`. It has shape `[batch_size]`.", "T1", true /*optional*/);
            schema.Input("initial_h",
                "Optional initial value of the hidden. If not specified - assumed "
                "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.", "T", true /*optional*/);
            schema.Output("Y",
                "A tensor that concats all the intermediate output values of the hidden."
                "It has shape `[seq_length, num_directions, batch_size, hidden_size]`.", "T");
            schema.Output("Y_h",
                "The last output value of the hidden. It has shape "
                "`[num_directions, batch_size, hidden_size]`.", "T");
            schema.Attr("direction", "Specify if the RNN is forward, reverse, or bidirectional. "
                "Must be one of forward (default), reverse, or bidirectional.",
                AttrType::AttributeProto_AttributeType_STRING);
            schema.Attr("hidden_size", "Number of neurons in the hidden layer",
                AttrType::AttributeProto_AttributeType_INT);
            schema.Attr("alpha",
                "Optional scaling values used by some activation functions.",
                AttrType::AttributeProto_AttributeType_FLOATS);
            schema.Attr("beta",
                "Optional scaling values used by some activation functions.",
                AttrType::AttributeProto_AttributeType_FLOATS);
            schema.TypeConstraint("T1", { "tensor(int32)" }, "Constrain seq_lens to integer tensor.");
            schema.TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                "Constrain input and output types to float tensors.");
        };
    }

    // TODO: An attribute "output_sequence" missing here per op specification doc.
    // Check with Radu/Sherlock on this later.
    REGISTER_OPERATOR_SCHEMA(RNN)
        .Description(R"DOC(
            Computes an one-layer simple RNN. This operator is usually supported
            via some custom implementation such as CuDNN.

            Notations:
            `X` - input tensor
            `i` - input gate
            `t` - time step (t-1 means previous time step)
            `Wi` - W parameter weight matrix for input gate
            `Ri` - R recurrence weight matrix for input gate
            `Wbi` - W parameter bias vector for input gate
            `Rbi` - R parameter bias vector for input gate
            `WBi` - W parameter weight matrix for backward input gate
            `RBi` - R recurrence weight matrix for backward input gate
            `WBbi` - WR bias vectors for backward input gate
            `RBbi` - RR bias vectors for backward input gate
            `ReLU(X)` - max(X, 0)
            `tanh(X)` - hyperbolic tangent of X
            `H` - Hidden state
            `num_directions` - 2 if direction == bidirectional else 1

            Equations:
              - Ht = Activation(Wi*Xt + Ri*Ht-1 + Wbi + Rbi)
            )DOC")
        .Input("W",
            "The weight tensor for input gate. Concatenation of `Wi` and `WBi` "
            "(if bidirectional). The tensor has shape "
            "`[num_directions, hidden_size, input_size]`.", "T")
        .Input("R",
            "The recurrence weight tensor. Concatenation of `Ri` and `RBi` "
            "(if bidirectional). The tensor has shape "
            "`[num_directions, hidden_size, hidden_size]`.", "T")
        .Input("B",
            "The bias tensor for input gate. Concatenation of `[Wbi, Rbi]` "
            "and `[WBbi, RBbi]` (if bidirectional). The tensor has shape "
            "`[num_directions, 2*hidden_size]`, Optional: If not specified - assumed "
            "to be 0.", "T",
            true)
        .Attr("activations", "One (or two if bidirectional) activation function for "
            "input gate. It must be one of tanh and ReLU. Default `tanh`.",
            AttrType::AttributeProto_AttributeType_STRINGS)
        .FillUsing(RNNDocGenerator());

    REGISTER_OPERATOR_SCHEMA(GRU)
        .Description(R"DOC(
            Computes an one-layer GRU. This operator is usually supported via some custom
            implementation such as CuDNN.

            Notations:
            `X` - input tensor
            `z` - update gate
            `r` - reset gate
            `h` - hidden gate
            `t` - time step (t-1 means previous time step)
            `W[zrh]` - W parameter weight matrix for update, reset, and hidden gates
            `R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates
            `Wb[zrh]` - W bias vectors for update, reset, and hidden gates
            `Rb[zrh]` - R bias vectors for update, reset, and hidden gates
            `WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates
            `RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates
            `WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates
            `RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates
            `tanh(X)` - hyperbolic tangent of X
            `sigmoid(X)` - 1 / (1 + e^-X)
            `H` - Hidden state
            `num_directions` - 2 if direction == bidirectional else 1

            Equations (GRU with default activations):
              - zt = sigmoid(Wz*Xt + Rz*Ht-1 + Wbz + Rbz)
              - rt = sigmoid(Wr*Xt + Rr*Ht-1 + Wbr + Rbr)
              - ht = tanh(Wh*Xt + rt*(Rh*Ht-1 + Rbh) + Wbh)
              - H = (1 - zt) (.) ht + it (.) Ht-1
            )DOC")
        .Input("W",
            "The weight tensor for the gates. Concatenation of `W[zrh]` and `WB[zrh]` "
            "(if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 3*hidden_size, input_size]`.", "T")
        .Input("R",
            "The recurrence weight tensor. Concatenation of `R[zrh]` and `RB[zrh]` "
            "(if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 3*hidden_size, hidden_size]`.", "T")
        .Input("B",
            "The bias tensor for the gates. Concatenation of `[Wb[zrh], Rb[zrh]]` and "
            "`[WBb[zrh], RBb[zrh]]` (if bidirectional) along dimension 0. This tensor "
            "has shape `[num_directions, 6*hidden_size]`. Optional: If not specified "
            "- assumed to be 0", "T",
            true /*optional*/)
        .Attr("activations", "A list of 3 (or 6 if bidirectional) activation functions "
            "for update, reset, and hidden gates. The activation functions must be "
            "one of sigmoid and tanh. See the equations for default.",
            AttrType::AttributeProto_AttributeType_STRINGS)
        .FillUsing(RNNDocGenerator());


    REGISTER_OPERATOR_SCHEMA(LSTM)
        .Description(R"DOC(
            Computes an one-layer LSTM. This operator is usually supported via some
            custom implementation such as CuDNN.

            Notations:
            `X` - input tensor
            `i` - input gate
            `o` - output gate
            `f` - forget gate
            `c` - cell gate
            `t` - time step (t-1 means previous time step)
            `W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates
            `R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates
            `Wb[iofc]` - W bias vectors for input, output, forget, and cell gates
            `Rb[iofc]` - R bias vectors for input, output, forget, and cell gates
            `P[iof]`  - P peephole weight vector for input, output, and forget gates
            `WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates
            `RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates
            `WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates
            `RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates
            `PB[iof]`  - P peephole weight vector for backward input, output, and forget gates
            `tanh(X)` - hyperbolic tangent of X
            `sigmoid(X)` - 1 / (1 + e^-X)
            `H` - Hidden state
            `num_directions` - 2 if direction == bidirectional else 1

            Equations (forward LSTM with default activations and peepholes):
              - it = sigmoid(Wi*Xt + Ri*Ht-1 + Pi (.) Ct-1 + Wbi + Rbi)
              - ft = sigmoid(Wf*Xt + Rf*Ht-1 + Pf (.) Ct-1 + Wbf + Rbf)
              - ct = tanh(Wc*Xt + Rc*Ht-1 + Wbc + Rbc)
              - Ct = ft (.) Ct-1 + it (.) ct
              - ot = sigmoid(Wo*Xt + Ro*Ht-1 + Po (.) Ct + Wbo + Rbo)
              - H = ot (.) tanh(Ct)
            )DOC")
        .Input("W",
            "The weight tensor for the gates. Concatenation of `W[zrh]` and `WB[zrh]` "
            "(if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 3*hidden_size, input_size]`.", "T")
        .Input("R",
            "The recurrence weight tensor. Concatenation of `R[zrh]` and `RB[zrh]` "
            "(if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 3*hidden_size, hidden_size]`.", "T")
        .Input("B",
            "The bias tensor for the gates. Concatenation of `[Wb[zrh], Rb[zrh]]` and "
            "`[WBb[zrh], RBb[zrh]]` (if bidirectional) along dimension 0. This tensor "
            "has shape `[num_directions, 6*hidden_size]`. Optional: If not specified "
            "- assumed to be 0", "T",
            true /*optional*/)
        .Attr("activations", "A list of 3 (or 6 if bidirectional) activation functions "
            "for update, reset, and hidden gates. The activation functions must be "
            "one of sigmoid and tanh. See the equations for default.",
            AttrType::AttributeProto_AttributeType_STRINGS)
        .Attr("clip", "Cell clip threshold. Clipping bounds the elements of a tensor "
            "in the range of [-threshold, +threshold] and is applied to the input "
            "of activations. No clip if not specified.",
            AttrType::AttributeProto_AttributeType_FLOAT)
        .Attr("input_forget", "Couple the input and forget gates if 1, default 0.",
            AttrType::AttributeProto_AttributeType_INT)
        .Input("initial_c",
            "Optional initial value of the cell. If not specified - assumed "
            "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.", "T",
            true /*optional*/)
        .Input("P",
            "The weight tensor for peepholes. Concatenation of `P[iof]` and "
            "`PB[iof]` (if bidirectional) along dimension 0. It has shape "
            "`[num_directions, 3*hidde_size]`. Optional: If not specified - "
            "assumed to be 0.", "T",
            true /*optional*/)
        .FillUsing(RNNDocGenerator());
}