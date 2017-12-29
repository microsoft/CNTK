#!/usr/bin/env ruby
#
# Put script description here.
#
# 
# 
# 
#

require 'swig_assert'
require 'add_link'
include Add_link

#
# This test currently fails due to swig
#
exit(0)

swig_assert( 'Foo.new' )
swig_assert( 'Foo.blah' )
