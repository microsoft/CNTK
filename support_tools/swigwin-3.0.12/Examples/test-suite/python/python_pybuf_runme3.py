# run:
#  python python_pybuf_runme3.py benchmark
# for the benchmark, other wise the test case will be run
import python_pybuf
import sys
if len(sys.argv) >= 2 and sys.argv[1] == "benchmark":
    # run the benchmark
    import time
    k = 1000000  # number of times to excute the functions

    t = time.time()
    a = bytearray(b'hello world')
    for i in range(k):
        pybuf.title1(a)
    print("Time used by bytearray:", time.time() - t)

    t = time.time()
    b = 'hello world'
    for i in range(k):
        pybuf.title2(b)
    print("Time used by string:", time.time() - t)
else:
    # run the test case
    buf1 = bytearray(10)
    buf2 = bytearray(50)

    pybuf.func1(buf1)
    assert buf1 == b'a' * 10

    pybuf.func2(buf2)
    assert buf2.startswith(b"Hello world!\x00")

    count = pybuf.func3(buf2)
    assert count == 10  # number of alpha and number in 'Hello world!'

    length = pybuf.func4(buf2)
    assert length == 12

    buf3 = bytearray(b"hello")
    pybuf.title1(buf3)
    assert buf3 == b'Hello'
