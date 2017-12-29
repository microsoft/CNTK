# Check std::vector and std::list behaves the same as Python iterable
# types (list)

from li_std_containers_int import *
import sys


def failed(a, b, msg):
    raise RuntimeError, msg + " " + str(list(a)) + " " + str(list(b))


def compare_sequences(a, b):
    if len(a) != len(b):
        failed(a, b, "different sizes")
    for i in range(len(a)):
        if a[i] != b[i]:
            failed(a, b, "elements are different")


def compare_containers(pythonlist, swigvector, swiglist):
    compare_sequences(pythonlist, swigvector)
    compare_sequences(pythonlist, swiglist)

# Check std::vector and std::list assignment behaves same as Python list
# assignment including exceptions


def container_insert_step(i, j, step, newval):
    ps = range(6)
    iv = vector_int(ps)
    il = list_int(ps)

    # Python slice
    try:
        if step == None:
            if j == None:
                ps[i] = newval
            else:
                ps[i:j] = newval
        else:
            if j == None:
                ps[i::step] = newval
            else:
                ps[i:j:step] = newval
        ps_error = None
    except ValueError, e:
        ps_error = e
    except IndexError, e:
        ps_error = e

    # std::vector<int>
    try:
        if step == None:
            if j == None:
                iv[i] = newval
            else:
                iv[i:j] = newval
        else:
            if j == None:
                iv[i::step] = newval
            else:
                iv[i:j:step] = newval
        iv_error = None
    except ValueError, e:
        iv_error = e
    except IndexError, e:
        iv_error = e

    # std::list<int>
    try:
        if step == None:
            if j == None:
                il[i] = newval
            else:
                il[i:j] = newval
        else:
            if j == None:
                il[i::step] = newval
            else:
                il[i:j:step] = newval
        il_error = None
    except ValueError, e:
        il_error = e
    except IndexError, e:
        il_error = e

    # Python 2.6 contains bug fixes in extended slicing syntax:
    # http://docs.python.org/2/whatsnew/2.6.html
    skip_check = ps_error != None and(
        iv_error == il_error == None) and step > 0 and (sys.version_info[0:2] < (2, 6))
    if not(skip_check):
        if not((type(ps_error) == type(iv_error)) and (type(ps_error) == type(il_error))):
            raise RuntimeError, "ValueError exception not consistently thrown: " + \
                str(ps_error) + " " + str(iv_error) + " " + str(il_error)

        compare_containers(ps, iv, il)


# Check std::vector and std::list delete behaves same as Python list
# delete including exceptions
def container_delete_step(i, j, step):
    ps = range(6)
    iv = vector_int(ps)
    il = list_int(ps)

    # Python slice
    try:
        if step == None:
            if j == None:
                del ps[i]
            else:
                del ps[i:j]
        else:
            if j == None:
                del ps[i::step]
            else:
                del ps[i:j:step]
        ps_error = None
    except ValueError, e:
        ps_error = e
    except IndexError, e:
        ps_error = e

    # std::vector<int>
    try:
        if step == None:
            if j == None:
                del iv[i]
            else:
                del iv[i:j]
        else:
            if j == None:
                del iv[i::step]
            else:
                del iv[i:j:step]
        iv_error = None
    except ValueError, e:
        iv_error = e
    except IndexError, e:
        iv_error = e

    # std::list<int>
    try:
        if step == None:
            if j == None:
                del il[i]
            else:
                del il[i:j]
        else:
            if j == None:
                del il[i::step]
            else:
                del il[i:j:step]
        il_error = None
    except ValueError, e:
        il_error = e
    except IndexError, e:
        il_error = e

    if not((type(ps_error) == type(iv_error)) and (type(ps_error) == type(il_error))):
        raise RuntimeError, "ValueError exception not consistently thrown: " + \
            str(ps_error) + " " + str(iv_error) + " " + str(il_error)

    compare_containers(ps, iv, il)


ps = [0, 1, 2, 3, 4, 5]

iv = vector_int(ps)
il = list_int(ps)

# slices
compare_containers(ps[0:0], iv[0:0], il[0:0])
compare_containers(ps[1:1], iv[1:1], il[1:1])
compare_containers(ps[1:3], iv[1:3], il[1:3])
compare_containers(ps[2:4], iv[2:4], il[2:4])
compare_containers(ps[0:3], iv[0:3], il[0:3])
compare_containers(ps[3:6], iv[3:6], il[3:6])
compare_containers(ps[3:10], iv[3:10], il[3:10])  # beyond end of range

# before beginning of range (negative indexing)
compare_containers(ps[-1:7], iv[-1:7], il[-1:7])
compare_containers(ps[-2:7], iv[-2:7], il[-2:7])
compare_containers(ps[-5:7], iv[-5:7], il[-5:7])
compare_containers(ps[-6:7], iv[-6:7], il[-6:7])

# before beginning of range (negative indexing, negative index is >
# container size)
compare_containers(ps[-7:7], iv[-7:7], il[-7:7])
compare_containers(ps[-100:7], iv[-100:7], il[-100:7])

compare_containers(ps[3:], iv[3:], il[3:])
compare_containers(ps[:3], iv[:3], il[:3])
compare_containers(ps[:], iv[:], il[:])
compare_containers(ps[-3:], iv[-3:], il[-3:])
compare_containers(ps[-7:], iv[-7:], il[-7:])
compare_containers(ps[:-1], iv[:-1], il[:-1])
compare_containers(ps[:-7], iv[:-7], il[:-7])

# step slicing
compare_containers(ps[1:5:1], iv[1:5:1], il[1:5:1])
compare_containers(ps[1:5:2], iv[1:5:2], il[1:5:2])
compare_containers(ps[1:5:3], iv[1:5:3], il[1:5:3])
compare_containers(ps[1:5:4], iv[1:5:4], il[1:5:4])
compare_containers(ps[1:6:5], iv[1:6:5], il[1:6:5])
compare_containers(ps[1:7:5], iv[1:7:5], il[1:7:5])
compare_containers(ps[-1:7:1], iv[-1:7:1], il[-1:7:1])
compare_containers(ps[-1:7:2], iv[-1:7:2], il[-1:7:2])
compare_containers(ps[-6:7:2], iv[-6:7:2], il[-6:7:2])
compare_containers(ps[-100:7:2], iv[-100:7:2], il[-100:7:2])
compare_containers(ps[::1], iv[::1], il[::1])
compare_containers(ps[::2], iv[::2], il[::2])

compare_containers(ps[::-1], iv[::-1], il[::-1])
compare_containers(ps[6::-1], iv[6::-1], il[6::-1])
compare_containers(ps[:-3:-1], iv[:-3:-1], il[:-3:-1])
compare_containers(ps[:-6:-1], iv[:-6:-1], il[:-6:-1])
compare_containers(ps[:-7:-1], iv[:-7:-1], il[:-7:-1])
compare_containers(ps[:-8:-1], iv[:-8:-1], il[:-8:-1])
compare_containers(ps[:-100:-1], iv[:-100:-1], il[:-100:-1])
compare_containers(ps[4:6:-1], iv[4:6:-1], il[4:6:-1])
compare_containers(ps[4:5:-1], iv[4:5:-1], il[4:5:-1])
compare_containers(ps[4:4:-1], iv[4:4:-1], il[4:4:-1])
compare_containers(ps[4:3:-1], iv[4:3:-1], il[4:3:-1])
compare_containers(ps[4:2:-1], iv[4:2:-1], il[4:2:-1])
compare_containers(ps[100:104:-1], iv[100:104:-1], il[100:104:-1])
compare_containers(ps[104:100:-1], iv[104:100:-1], il[104:100:-1])
compare_containers(ps[-100:-104:-1], iv[-100:-104:-1], il[-100:-104:-1])
compare_containers(ps[-104:-100:-1], iv[-104:-100:-1], il[-104:-100:-1])
compare_containers(ps[::-2], iv[::-2], il[::-2])
compare_containers(ps[::-3], iv[::-3], il[::-3])
compare_containers(ps[::-4], iv[::-4], il[::-4])
compare_containers(ps[::-5], iv[::-5], il[::-5])


# insert sequences (growing, shrinking and staying same size)
for start in [-102, -100, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 100, 102]:
    # single element set/replace
    container_insert_step(start, None, None, 111)
    for end in [-102, -100, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 100, 102]:
        container_insert_step(
            start, end, None, [111, 222, 333, 444, 555, 666, 777])
        container_insert_step(start, end, None, [111, 222, 333, 444, 555, 666])
        container_insert_step(start, end, None, [111, 222, 333, 444, 555])
        container_insert_step(start, end, None, [111, 222, 333, 444])
        container_insert_step(start, end, None, [111, 222, 333])
        container_insert_step(start, end, None, [111, 222])
        container_insert_step(start, end, None, [111])
        container_insert_step(start, end, None, [])

# delete sequences (growing, shrinking and staying same size)
for start in [-102, -100, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 100, 102]:
    # single element delete
    container_delete_step(start, None, None)
    for end in [-102, -100, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 100, 102]:
        container_delete_step(start, end, None)
        for step in range(-7, 7):
            container_delete_step(start, end, step)

ps = range(6)
iv = vector_int(ps)
il = list_int(ps)
del ps[:]
del iv[:]
del il[:]
compare_containers(ps, iv, il)

for end in range(7):
    for step in range(-7, 7):
        for start in range(7):
            container_insert_step(
                start, end, step, [111, 222, 333, 444, 555, 666, 777])
            container_insert_step(
                start, end, step, [111, 222, 333, 444, 555, 666])
            container_insert_step(start, end, step, [111, 222, 333, 444, 555])
            container_insert_step(start, end, step, [111, 222, 333, 444])
            container_insert_step(start, end, step, [111, 222, 333])
            container_insert_step(start, end, step, [111, 222])
            container_insert_step(start, end, step, [111])
            container_insert_step(start, end, step, [])

try:
    x = iv[::0]
    raise RuntimeError("Zero step not caught")
except ValueError:
    pass
