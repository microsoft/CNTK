
from overload_numeric import *
import math

nums = Nums()
limits = Limits()


def check(got, expected):
    if got != expected:
        raise RuntimeError("got: " + got + " expected: " + expected)

check(nums.over(0), "signed char")
check(nums.over(0.0), "float")

check(nums.over(limits.schar_min()), "signed char")
check(nums.over(limits.schar_max()), "signed char")

check(nums.over(limits.schar_min() - 1), "short")
check(nums.over(limits.schar_max() + 1), "short")
check(nums.over(limits.shrt_min()), "short")
check(nums.over(limits.shrt_max()), "short")

check(nums.over(limits.shrt_min() - 1), "int")
check(nums.over(limits.shrt_max() + 1), "int")
check(nums.over(limits.int_min()), "int")
check(nums.over(limits.int_max()), "int")

check(nums.over(limits.flt_min()), "float")
check(nums.over(limits.flt_max()), "float")

check(nums.over(limits.flt_max() * 10), "double")
check(nums.over(-limits.flt_max() * 10), "double")
check(nums.over(limits.dbl_max()), "double")
check(nums.over(-limits.dbl_max()), "double")

check(nums.over(float("inf")), "float")
check(nums.over(float("-inf")), "float")
check(nums.over(float("nan")), "float")

# Just check if the following are accepted without exceptions being thrown
nums.doublebounce(float("inf"))
nums.doublebounce(float("-inf"))
nums.doublebounce(float("nan"))
