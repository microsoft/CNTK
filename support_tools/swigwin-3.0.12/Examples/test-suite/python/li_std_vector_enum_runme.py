import li_std_vector_enum


def check(a, b):
    if (a != b):
        raise RuntimeError("Not equal: ", a, b)

ev = li_std_vector_enum.EnumVector()

check(ev.nums[0], 10)
check(ev.nums[1], 20)
check(ev.nums[2], 30)

it = ev.nums.iterator()
v = it.value()
check(v, 10)
it.next()
v = it.value()
check(v, 20)

expected = 10
for val in ev.nums:
    check(val, expected)
    expected += 10
