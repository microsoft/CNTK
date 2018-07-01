@REM ==============================================================================
@REM Copyright (c) Microsoft. All rights reserved.
@REM Licensed under the MIT license. See LICENSE.md file in the project root
@REM for full license information.
@REM ==============================================================================

setlocal

@REM Trick cntkpy35.bat into believing we're legitimate
set CMDCMDLINE="%COMSPEC%" &@REM do not delete the space to the left

call "%~1"

for /f "delims=" %%i in ('python -c "import cntk, os, sys; sys.stdout.write(os.path.dirname(os.path.abspath(cntk.__file__)))"') do set MODULE_DIR=%%i
if errorlevel 1 exit /b 1

set TEST_LIST=not op2cntk_test and not cntk_utils_test and not policy_gradient_test and not qlearning_test ^
and not factorization_test and not debug_test and not userlearner_test and not evaluator_distributed_test ^
and not evaluator_test and not utils_test and not io_tests and not blocks and not higher_order_layers ^
and not layers and not __init__ and not bmuf_metrics_aggregation_test and not distributed_multi_learner_test ^
and not learner_test and not progress_print_test and not cosine_distance_test and not h_softmax_test ^
and not losses_test and not metrics_test and not functions and not assign_test and not block_test ^
and not combine_test and not evaluation_test and not fp16_test and not free_static_axis_test ^
and not function_tests and not kernel_test and not linear_test and not non_diff_test and not recurrent_test ^
and not reshaping_test and not sequence_test and not sparse_test and not stop_gradient_test ^
and not userfunction_complex_test and not userfunction_test and not random_ops_test and not function_test ^
and not onnx_format_test and not onnx_op_test and not persist_test and not tensor_test and not value_test ^
and not variables_test and not distributed_test and not trainer_test and not training_session and not misc_test

where cntk && ^
pytest %MODULE_DIR% -k "%TEST_LIST%"

exit /b %ERRORLEVEL%
