#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'


require 'bools'

# bool constant check
if (Bools::Constbool != false)
    print "Runtime test 1 failed\n"
    exit 1
end

# bool variables check
if (Bools.bool1 != true)
    print "Runtime test 2 failed\n"
    exit 1
end

if (Bools.bool2 != false)
    print "Runtime test 3 failed\n"
    exit 1
end

if (Bools.value(Bools.pbool) != Bools.bool1)
    print "Runtime test 4 failed\n"
    exit 1
end

if (Bools.value(Bools.rbool) != Bools.bool2)
    print "Runtime test 5 failed\n"
    exit 1
end

if (Bools.value(Bools.const_pbool) != Bools.bool1)
    print "Runtime test 6 failed\n"
    exit 1
end

if (Bools.const_rbool != Bools.bool2)
    print "Runtime test 7 failed\n"
    exit 1
end

# bool functions check
if (Bools.bo(false) != false)
    print "Runtime test 8 failed\n"
    exit 1
end

if (Bools.bo(true) != true)
    print "Runtime test 9 failed\n"
    exit 1
end

if (Bools.value(Bools.rbo(Bools.rbool)) != Bools.value(Bools.rbool))
    print "Runtime test 10 failed\n"
    exit 1
end

if (Bools.value(Bools.pbo(Bools.pbool)) != Bools.value(Bools.pbool))
    print "Runtime test 11 failed\n"
    exit 1
end

if (Bools.const_rbo(Bools.const_rbool) != Bools.const_rbool)
    print "Runtime test 12 failed\n"
    exit 1
end

if (Bools.value(Bools.const_pbo(Bools.const_pbool)) != Bools.value(Bools.const_pbool))
    print "Runtime test 13 failed\n"
    exit 1
end

