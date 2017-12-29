from li_cwstring import *

if count(u"ab\0ab\0ab\0", 0) != 3:
    raise RuntimeError

if test1() != u"Hello World":
    raise RuntimeError

if test2() != u" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_":
    raise RuntimeError

if test3("hello") != u"hello-suffix":
    raise RuntimeError

if test4("hello") != u"hello-suffix":
    raise RuntimeError

if test5(4) != u'xxxx':
    raise RuntimeError

if test6(10) != u'xxxxx':
    raise RuntimeError

if test7() != u"Hello world!":
    raise RuntimeError

if test8() != u" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_":
    raise RuntimeError
