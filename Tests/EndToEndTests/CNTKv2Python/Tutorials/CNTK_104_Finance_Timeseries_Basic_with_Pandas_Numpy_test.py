# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import re

abs_path = os.path.dirname(os.path.abspath(__file__))
notebook = os.path.join(abs_path, "..", "..", "..", "..", "Tutorials", "CNTK_104_Finance_Timeseries_Basic_with_Pandas_Numpy.ipynb")

def test_cntk_104_finance_timeseries_basic_with_pandas_numpy_noErrors(nb):
    errors = [output for cell in nb.cells if 'outputs' in cell
              for output in cell['outputs'] if output.output_type == "error"]
    print(errors)
    assert errors == []

expectedEvalErrorUpper = 0.55
expectedEvalErrorLower = 0.45

def test_cntk_104_finance_timeseries_basic_with_pandas_numpy_trainerror(nb):
    testCell = [cell for cell in nb.cells
                if cell.cell_type == 'code' and re.search('# Repeatable factor ', cell.source)]
    assert (float((testCell[0].outputs[0])['text']) < expectedEvalErrorUpper) or (float((testCell[0].outputs[0])['text']) > expectedEvalErrorLower)