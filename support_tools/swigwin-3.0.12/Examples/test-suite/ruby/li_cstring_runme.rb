#!/usr/bin/env ruby
#
# Test for li_cstring.i
#

require 'swig_assert'
require 'li_cstring'

include Li_cstring

swig_assert_each_line <<EOF
count("hello", 'l'[0]) == 2
test1 == 'Hello World'
test2
test3('hello') == 'hello-suffix'
test4('hello') == 'hello-suffix'
test5(5) == 'xxxxx'
test6(6) == 'xxx'
test7    == 'Hello world!'
test8    == (32..32+63).map {|x| x.chr }.join
EOF

