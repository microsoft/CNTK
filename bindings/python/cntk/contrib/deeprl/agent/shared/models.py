import numpy as np
from cntk.layers import AveragePooling, Dense, For, Sequential
from cntk.losses import squared_error
from cntk.ops import input_variable, placeholder, relu

import ast


class Models:
    """A set of predefined models to approximate Q or log of pi (policy).

    The loss function needs to be 'cross_entropy_with_softmax' for policy
    gradient methods.
    """

    @staticmethod
    def feedforward_network(shape_of_inputs,
                            number_of_outputs,
                            model_hidden_layers,
                            loss_function=None,
                            use_placeholder_for_input=False):
        """Feedforward network to approximate Q or log of pi.

        Args:
            shape_of_inputs: tuple of array (input) dimensions.
            number_of_outputs: dimension of output, equals the number of
                possible actions.
            model_hidden_layers: list of integers representing number of nodes
                in each hidden layer.
            loss_function: if not specified, use squared loss by default.
            use_placeholder_for_input: if true, inputs have to be replaced
                later with actual input_variable.

        Returns: a Python dictionary with string valued keys including
            'inputs', 'outputs', 'loss' and 'f'.
        """
        # input/output
        inputs = placeholder(shape=shape_of_inputs) \
            if use_placeholder_for_input \
            else input_variable(shape=shape_of_inputs, dtype=np.float32)
        outputs = input_variable(shape=(number_of_outputs,), dtype=np.float32)

        # network structure
        hidden_layers = ast.literal_eval(model_hidden_layers)
        f = Sequential([
            For(range(len(hidden_layers)),
                lambda h: Dense(hidden_layers[h], activation=relu)),
            Dense(number_of_outputs, activation=None)
        ])(inputs)

        if loss_function is None:
            loss = squared_error(f, outputs)
        else:
            loss = loss_function(f, outputs)

        return {
            'inputs': inputs,
            'outputs': outputs,
            'f': f,
            'loss': loss
        }

    @staticmethod
    def dueling_network(shape_of_inputs,
                        number_of_outputs,
                        model_hidden_layers,
                        loss_function=None,
                        use_placeholder_for_input=False):
        """Dueling network to approximate Q function.

        See paper at https://arxiv.org/pdf/1511.06581.pdf.

        Args:
            shape_of_inputs: tuple of array (input) dimensions.
            number_of_outputs: dimension of output, equals the number of
                possible actions.
            model_hidden_layers: in the form of "[comma-separated integers,
                [comma-separated integers], [comma-separated integers]]". Each
                integer is the number of nodes in a hidden layer.The
                first set of integers represent the shared component in dueling
                network. The second set correponds to the state value function
                V and the third set correponds to the advantage function A.
            loss_function: if not specified, use squared loss by default.
            use_placeholder_for_input: if true, inputs have to be replaced
                later with actual input_variable.

        Returns: a Python dictionary with string-valued keys including
            'inputs', 'outputs', 'loss' and 'f'.
        """
        # input/output
        inputs = placeholder(shape=shape_of_inputs) \
            if use_placeholder_for_input \
            else input_variable(shape=shape_of_inputs, dtype=np.float32)
        outputs = input_variable(shape=(number_of_outputs,), dtype=np.float32)

        # network structure
        shared_hidden_layers, v_hidden_layers, a_hidden_layers =\
            Models._parse_dueling_network_structure(model_hidden_layers)
        # shared layers
        s = For(range(len(shared_hidden_layers)),
                lambda h: Dense(shared_hidden_layers[h], activation=relu))(inputs)
        # Value function
        v = Sequential([
            For(range(len(v_hidden_layers)),
                lambda h: Dense(v_hidden_layers[h], activation=relu)),
            Dense(1, activation=None)
        ])(s)
        # Advantage function
        a = Sequential([
            For(range(len(a_hidden_layers)),
                lambda h: Dense(a_hidden_layers[h], activation=relu)),
            Dense(number_of_outputs, activation=None)
        ])(s)
        # Q = V + A - avg(A)
        # TODO(maoyi): GlobalAveragePooling() may be a better alternative, but
        # it gives segmentation fault for now.
        avg_a = AveragePooling((number_of_outputs,))(a)
        q = v + a - avg_a

        if loss_function is None:
            loss = squared_error(q, outputs)
        else:
            loss = loss_function(q, outputs)

        return {
            'inputs': inputs,
            'outputs': outputs,
            'f': q,
            'loss': loss
        }

    @staticmethod
    def _parse_dueling_network_structure(hidden_layers_str):
        hidden_layers = ast.literal_eval(hidden_layers_str)

        if not (
            len(hidden_layers) > 2
                and isinstance(hidden_layers[-1], list)
                and isinstance(hidden_layers[-2], list)):
            raise ValueError('Invalid dueling network structure.')

        return\
            Models._remove_none_elements_from_list(hidden_layers[:-2]),\
            Models._remove_none_elements_from_list(hidden_layers[-2]),\
            Models._remove_none_elements_from_list(hidden_layers[-1])

    @staticmethod
    def _remove_none_elements_from_list(value_list):
        return [e for e in value_list if e is not None]
