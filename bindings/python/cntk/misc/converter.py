# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
import cntk as C

def convert(root_func, filter, converter):
    '''
    Clones the graph underlying root_func and in the clone substitutes
    all Functions obtained by applying 'filter', with a new Function obtained by calling the specified 'converter'

    Args:
        root_func: a root function of a graph to be cloned and converted
        filter: a lambda for filtering out the Functions to be converted
        converter: a lambda for obtaining the substitute for each of the Functions to be converted
    Returns:
        Cloned and converted Function (graph)
    '''
    # recursively convert for blocks in root_func
    blocks = C.logging.graph.depth_first_search(root_func, lambda x : type(x) == C.Function and x.root_function.is_block, depth = 0)
    for i in range(len(blocks)):
        # search for blocks again in case block input/output has been modified
        blocks1 = C.logging.graph.depth_first_search(root_func, lambda x : type(x) == C.Function and x.root_function.is_block, depth = 0)
        block = blocks1[i] # assuming depth_first_search order to be stable, so use the old index on new search results
        block_root = C.as_composite(block.block_root)
        new_block_root = convert(block_root, filter, converter)
        if new_block_root != block_root:
            block_arguments_mapping = dict(block.block_arguments_mapping)
            new_block_arguments_mapping = []
            for arg, new_arg in zip(block_root.arguments, new_block_root.arguments):
                new_block_arguments_mapping += [(new_arg, block_arguments_mapping[arg])]
            new_block = C.as_block(new_block_root, new_block_arguments_mapping, block.op_name, block.name)
            if all([x not in root_func.outputs for x in block.outputs]) or all([x in block.outputs for x in root_func.outputs]):
                root_func = root_func.clone(C.CloneMethod.share, dict(zip(block.outputs, new_block.outputs)))
            else:
                new_outputs = [new_block.outputs[block.outputs.index(x)] if x in block.outputs else None for x in root_func.outputs]
                root_func_nonreplaced = C.combine([x for x in root_func.outputs if x not in block.outputs])
                root_func_nonreplaced_clone = root_func_nonreplaced.clone(C.CloneMethod.share, dict(zip(block.outputs, new_block.outputs)))
                idx = 0
                for nonreplaced_output in root_func_nonreplaced_clone.outputs:
                    while new_outputs[idx]:
                        idx += 1
                    new_outputs[idx] = nonreplaced_output
                root_func = C.combine(new_outputs)

    # replace all Function instances under root_func that pass the specified 'filter'
    functions_to_convert = C.logging.graph.depth_first_search(root_func, filter, depth = 0)
    for i in range(len(functions_to_convert)):
        # The graph could be modified already by this function, so we need to rescan to the new set.
        functions_to_convert1 = C.logging.graph.depth_first_search(root_func, filter, depth = 0)
        # We are using a filter passed in by the caller. So once a function is converted, we may not
        # get the same number of functions again, so we need to use correct index depending on the new size.
        index = 0
        if len(functions_to_convert) > len(functions_to_convert1):
            assert(len(functions_to_convert) - len(functions_to_convert1) == i) # Only one conversion at a time.
            # index = 0 will work for this case, we are picking the first function from the new list.
        elif len(functions_to_convert) == len(functions_to_convert1):
            index = i # here we pick the current index of the for loop.
        else:
            raise RuntimeError("The conversion adds another possible conversion(s). Stopping infinite conversions.")

        function_to_convert = functions_to_convert1[index]
        converted = converter(function_to_convert)

        if not function_to_convert.output in root_func.outputs:            
            root_func = root_func.clone(C.CloneMethod.share, {function_to_convert.output : converted.output})
        else:
            # if cudnn_rnn output is the root_func output, just use converted as root_func and no clone needed
            if len(root_func.outputs) > 1:
                root_func = C.combine([converted if x == function_to_convert.output else x for x in root_func.outputs])
            else:
                root_func = converted

    return root_func
