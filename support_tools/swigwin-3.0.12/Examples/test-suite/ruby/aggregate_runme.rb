#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'aggregate'

include Aggregate

# Confirm that move() returns correct results under normal use
result = move(UP)
raise RuntimeError unless (result == UP)

result = move(DOWN)
raise RuntimeError unless (result == DOWN)

result = move(LEFT)
raise RuntimeError unless (result == LEFT)

result = move(RIGHT)
raise RuntimeError unless (result == RIGHT)

# Confirm that it raises an exception when the contract is violated
failed = false
begin
  move(0)
rescue RuntimeError
  failed = true
end
raise RuntimeError unless failed

