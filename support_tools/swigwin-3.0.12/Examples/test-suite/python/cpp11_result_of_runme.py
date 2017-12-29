import cpp11_result_of

result = cpp11_result_of.test_result(cpp11_result_of.SQUARE, 3.0)
if result != 9.0:
    raise RuntimeError, "test_result(square, 3.0) is not 9.0. Got: " + str(
        result)

result = cpp11_result_of.test_result_alternative1(cpp11_result_of.SQUARE, 3.0)
if result != 9.0:
    raise RuntimeError, "test_result_alternative1(square, 3.0) is not 9.0. Got: " + str(
        result)
