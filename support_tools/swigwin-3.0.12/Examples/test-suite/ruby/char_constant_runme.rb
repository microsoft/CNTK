#!/usr/bin/env ruby
#Encoding: ASCII-8BIT
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'char_constant'


if Char_constant::CHAR_CONSTANT != 'x'
  raise RuntimeError, "Invalid value for CHAR_CONSTANT."
end
  
if Char_constant::STRING_CONSTANT != 'xyzzy'
  raise RuntimeError, "Invalid value for STRING_CONSTANT."
end

if Char_constant::ESC_CONST != "\001"
  raise RuntimeError, "Invalid value for ESC_CONST."
end

if Char_constant::NULL_CONST != "\000"
  raise RuntimeError, "Invalid value for NULL_CONST."
end

if Char_constant::SPECIALCHAR != "\341"  #'á'
  raise RuntimeError, "Invalid value for SPECIALCHAR."
end

