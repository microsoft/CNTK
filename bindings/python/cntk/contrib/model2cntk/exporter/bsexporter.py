# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
import os
import shutil

from exporter import baseexporter
from unimodel import cntkmodel


# TODO: Enable keep style
def refresh_bs_style_list(input_attributes, name=None):  # , keep_style=False):
    various_str = str(input_attributes)
    if isinstance(input_attributes, bool):
        various_str = various_str.lower()
    if isinstance(input_attributes, list):
        various_str = various_str.replace('[', '(')
        various_str = various_str.replace(']', ')')
        various_str = various_str.replace(', ', ':')
        various_str = various_str.replace(',', ':')
    return various_str if not name else '='.join((name, various_str))


def load_batch_norm_attributes(layer_parameters):
    attributes = list()
    attributes.append(refresh_bs_style_list(2, 'spatialRank'))
    attributes.append(refresh_bs_style_list(layer_parameters.scale_setting, 'scaleSetting'))
    attributes.append(refresh_bs_style_list(layer_parameters.bias_setting, 'biasSetting'))
    return attributes


# TODO: To support different initializer
def load_convolution_attributes(layer_parameters):
    attributes = list()
    attributes.append(refresh_bs_style_list(layer_parameters.output))
    attributes.append(refresh_bs_style_list(layer_parameters.kernel))
    attributes.append(refresh_bs_style_list(layer_parameters.need_bias, 'bias'))
    attributes.append(refresh_bs_style_list(layer_parameters.auto_pad, 'pad'))
    attributes.append(refresh_bs_style_list(layer_parameters.lower_pad, 'lowerPad'))
    attributes.append(refresh_bs_style_list(layer_parameters.stride, 'stride'))
    attributes.append(refresh_bs_style_list(layer_parameters.scale_setting, 'weightSetting'))
    attributes.append(refresh_bs_style_list(layer_parameters.bias_setting, 'biasSetting'))
    return attributes


def load_pooling_attributes(layer_parameters):
    attributes = list()
    attributes.append(refresh_bs_style_list('\"max\"' if not layer_parameters.pooling_type else '\"average\"'))
    attributes.append(refresh_bs_style_list(layer_parameters.kernel))
    attributes.append(refresh_bs_style_list(layer_parameters.auto_pad, 'pad'))
    attributes.append(refresh_bs_style_list(layer_parameters.lower_pad, 'lowerPad'))
    attributes.append(refresh_bs_style_list(layer_parameters.stride, 'stride'))
    return attributes


def load_linear_attributes(layer_parameters):
    attributes = list()
    attributes.append(refresh_bs_style_list(layer_parameters.num_output))
    attributes.append(refresh_bs_style_list(layer_parameters.scale_setting, 'weightSetting'))
    attributes.append(refresh_bs_style_list(layer_parameters.bias_setting, 'biasSetting'))
    return attributes


def load_splice_attributes(layer_parameters):
    attributes = list()
    attributes.append(refresh_bs_style_list(4 - layer_parameters.axis, 'axis'))  # TODO: Dangerous channel convert
    return attributes

EXPORT_LAYER_DEF = {
    cntkmodel.CntkLayerType.convolution: ('CB_ConvolutionLayer', load_convolution_attributes),
    cntkmodel.CntkLayerType.pooling: ('CB_PoolingLayer', load_pooling_attributes),
    cntkmodel.CntkLayerType.batch_normalization: ('CB_BatchNormalizationLayer', load_batch_norm_attributes),
    cntkmodel.CntkLayerType.dense: ('CB_LinearLayer', load_linear_attributes),
    cntkmodel.CntkLayerType.splice: ('CB_Splice', load_splice_attributes),
    cntkmodel.CntkLayerType.plus: ('Plus', None),
    cntkmodel.CntkLayerType.relu: ('ReLU', None),
    cntkmodel.CntkLayerType.cross_entropy_with_softmax: ('CrossEntropyWithSoftmax', None),
    cntkmodel.CntkLayerType.classification_error: ('ClassificationError', None),
    cntkmodel.CntkLayerType.dropout: ('Dropout', None),
}

CONCAT_INPUT_LAYER_LIST = [
    cntkmodel.CntkLayerType.splice,
]

EXPORT_ROOTS = ['root_nodes', 'softmax', 'classification', 'output']

TEMPLATE_LOCATION = os.path.join(sys.path[0], 'exporter', 'bs_template')
SHARE_NETWORK_PATH = os.path.join(TEMPLATE_LOCATION, 'network.bs')
TEMPLATE_NETWORK_PATH = os.path.join(TEMPLATE_LOCATION, 'template_network.bs')
TEMPLATE_SCRIPT_PATH = os.path.join(TEMPLATE_LOCATION, 'template_script.cntk')


class ExportRootDefinition(object):
    def __init__(self):
        self.root_nodes = None
        self.softmax = None
        self.classification = None
        self.output = None


class BsExporter(baseexporter.BaseExporter):

    def __init__(self, uni_model_desc):
        self._uni_model_desc = uni_model_desc
        self._export_roots_def = ExportRootDefinition()

    def export_scripts(self, export_path):
        export_dir_path = os.path.join(export_path, 'BS_EXPORT')
        # build the export folder
        sys.stdout.write('Re-direct the exporter files into %s\n' % export_dir_path)
        if os.path.isdir(export_dir_path):
            sys.stdout.write('Erase previous export files...\n')
            shutil.rmtree(export_dir_path)
        os.mkdir(export_dir_path)

        # copy the common macros
        export_share_macro_path = os.path.join(export_dir_path, 'network.bs')
        shutil.copy(SHARE_NETWORK_PATH, export_share_macro_path)

        model_name = self._uni_model_desc.model_name
        # build the network script
        export_network_path = os.path.join(export_dir_path, model_name + '.bs')
        sys.stdout.write('Export network to file: %s' % export_network_path)
        shutil.copy(TEMPLATE_NETWORK_PATH, export_network_path)
        self.export_network(export_network_path)
        # build the trainer script
        # TODO: Trainer
        export_trainer_path = os.path.join(export_dir_path, model_name + '.cntk')
        sys.stdout.write('Export training script to file: %s\n' % export_trainer_path)
        shutil.copy(TEMPLATE_SCRIPT_PATH, export_trainer_path)
        self.export_trainer(export_trainer_path)

    def export_trainer(self, export_path):
        solver = self._uni_model_desc.solver
        template_file = open(export_path, 'r')
        trainer_file = open('tmp', 'w')

        self.generate_roots_definition()

        for line in template_file.readlines():
            # TODO: Replace with regex equation
            start_pos = line.find('@')
            end_pos = line.find('@', start_pos + 1)
            if start_pos == -1 or start_pos == end_pos:
                trainer_file.write(line)
                continue
            front_seg = line[:start_pos]
            end_seg = line[end_pos + 1:]

            replace_seg = line[start_pos + 1: end_pos].lower()
            try:
                # Search segments in trainer
                replace_res = getattr(solver, replace_seg)
            except AttributeError:
                if replace_seg in EXPORT_ROOTS:
                    replace_res = getattr(self._export_roots_def, replace_seg)
                elif replace_seg == 'network_name':
                    replace_res = self._uni_model_desc.model_name
                else:
                    replace_res = 'NOT_FOUND_SEGMENT'
            if replace_res is None:
                replace_res = 'NOT_IMPLEMENTED_FROM_SCRIPT'
            if isinstance(replace_res, list):
                for single_out in replace_res:
                    trainer_file.write(''.join((front_seg, str(single_out), end_seg)))
            else:
                trainer_file.write(''.join((front_seg, str(replace_res), end_seg)))

        # file copy and remove
        template_file.close()
        trainer_file.close()
        os.remove(export_path)
        shutil.copy('tmp', export_path)
        os.remove('tmp')

    def export_network(self, export_path=''):
        network_script = open(export_path, 'a')

        # starting the input definition
        # TODO: using OI name instead of Op_name
        network_inputs = []
        for data_provide in self._uni_model_desc.data_provider:
            network_inputs.append(data_provide.op_name)
        network_script.write('model (%s) = {\n' % ', '.join(network_inputs))

        for layer_name in self._uni_model_desc.cntk_sorted_layers:
            layer_def = self._uni_model_desc.cntk_layers[layer_name]
            components = list()  # normally, four components: outputs/functions/parameters/inputs
            # indent
            components.append(' ' * 3)
            # attach output
            components.append(', '.join(layer_def.outputs))
            # equation
            components.append('=')
            # attach function
            components.append(EXPORT_LAYER_DEF[layer_def.op_type][0])
            # attach parameters
            parameters_line = BsExporter.attach_layer_parameters(layer_def.op_type, layer_def.parameters)
            if parameters_line:
                components.append(parameters_line.join(('{', '}')))
            # attach input
            components.append(BsExporter.attach_input_list(layer_def.op_type, layer_def.inputs).join(('(', ')')))

            # append line into script
            network_script.write(' '.join(components))
            network_script.write('\n')

        # ending mark
        network_script.write('}\n')
        network_script.close()

    def generate_roots_definition(self):
        model_prefix = 'z.'
        root_inputs = set()
        classification_names = []
        softmax_names = []
        roots_expression = []
        for root in self._uni_model_desc.cntk_model_roots:
            root_inputs.add(model_prefix + root.inputs[0])
            expression_components = list()
            expression_components.append(root.outputs[0])
            expression_components.append('=')
            expression_components.append(EXPORT_LAYER_DEF[root.op_type][0])
            if len(root_inputs) == 2:
                input_def = ', '.join((model_prefix + root.inputs[0], root.inputs[1]))
            else:
                input_def = ', '.join((model_prefix + root.inputs[0], 'label'))
            if root.op_type == cntkmodel.CntkLayerType.cross_entropy_with_softmax:
                softmax_names.append(root.op_name)
            else:
                classification_names.append(root.op_name)
                input_def = ', '.join((input_def, '='.join(('topN', str(root.parameters.top_n)))))
            expression_components.append(input_def.join(('(', ')')))
            roots_expression.append(' '.join(expression_components))
        self._export_roots_def.classification = ':'.join(classification_names).join(('(', ')'))
        self._export_roots_def.softmax = ':'.join(softmax_names).join(('(', ')'))
        self._export_roots_def.output = ':'.join(root_inputs).join(('(', ')'))
        self._export_roots_def.root_nodes = roots_expression

    @staticmethod
    def attach_input_list(cntk_layer_type, inputs_name):
        if cntk_layer_type in CONCAT_INPUT_LAYER_LIST:
            input_command = ':'.join(inputs_name)
            input_command = input_command.join(('(', ')'))
            return input_command
        else:
            input_command = ', '.join(inputs_name)
            return input_command

    @staticmethod
    def attach_layer_parameters(cntk_layer_type, layer_parameters):
        try:
            return ', '.join(EXPORT_LAYER_DEF[cntk_layer_type][1](layer_parameters))
        except TypeError:
            return None
