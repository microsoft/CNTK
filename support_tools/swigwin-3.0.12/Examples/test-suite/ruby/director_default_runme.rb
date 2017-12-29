#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'director_default'



a = Director_default::Foo.new 1
a = Director_default::Foo.new 

a.Msg 
a.Msg "hello"
a.GetMsg
a.GetMsg "hello"

a = Director_default::Bar.new 1
a = Director_default::Bar.new 

a.Msg 
a.Msg "hello"
a.GetMsg
a.GetMsg "hello"
