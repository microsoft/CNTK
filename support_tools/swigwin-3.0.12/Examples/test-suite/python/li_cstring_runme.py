from li_cstring import *


if count("ab\0ab\0ab\0", 0) != 3:
    raise RuntimeError

if test1() != "Hello World":
    raise RuntimeError

if test2() != " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_":
    raise RuntimeError

if test3("hello") != "hello-suffix":
    print test3("hello")
    raise RuntimeError

if test4("hello") != "hello-suffix":
    print test4("hello")
    raise RuntimeError

if test5(4) != 'xxxx':
    raise RuntimeError

if test6(10) != 'xxxxx':
    raise RuntimeError

if test7() != "Hello world!":
    raise RuntimeError

if test8() != " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_":
    raise RuntimeError
