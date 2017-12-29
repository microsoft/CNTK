from li_std_pair_using import *

one_tuple = ("one", "numero uno")
one = StringStringPair(one_tuple)
two_tuple = ("two", 2)
two = StringIntPair(two_tuple)

if bounce(one) != one_tuple:
    raise RuntimeError
