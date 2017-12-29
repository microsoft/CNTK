#!/usr/bin/env ruby
#
# Put script description here.
#

require 'swig_assert'
require 'integers'
include Integers

swig_assert_each_line <<EOF
signed_char_identity(-3)   == -3
unsigned_char_identity(5)  == 5
signed_short_identity(-3)  == -3
unsigned_short_identity(5) == 5
signed_int_identity(-3)  == -3
unsigned_int_identity(5) == 5
signed_long_identity(-3)  == -3
unsigned_long_identity(5) == 5
signed_long_long_identity(-3)  == -3
unsigned_long_long_identity(5) == 5
EOF
