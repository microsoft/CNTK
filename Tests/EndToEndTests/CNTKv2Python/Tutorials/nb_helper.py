# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

def get_output_stream_from_cell(cell, stream_name='stdout'):
    return "\n".join([output.text for output in cell.get('outputs', [])
        if output.output_type == 'stream' and output.name == stream_name])
