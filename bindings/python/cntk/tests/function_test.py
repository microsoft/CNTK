# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk import *

def test_outputs():
    fwd_state = placeholder_variable("placeholder")
    prev_state = past_value(fwd_state, name="prev_state")
    z = abs(prev_state, "abs")
    output = z.output
    z = z.replace_placeholders({fwd_state: z.output})

    fwd_state = None
    prev_state = None
    z = None

    for arg in output.owner.arguments:
        print("Argument name: {}, argument owner name {}".format(arg.name, arg.owner.name))

