#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'reference_global_vars'

# primitive reference variables
Reference_global_vars.var_bool = Reference_global_vars.createref_bool(true)
if (Reference_global_vars.value_bool(Reference_global_vars.var_bool) != true)
    print "Runtime error test 1\n"
    exit 1
end

Reference_global_vars.var_bool = Reference_global_vars.createref_bool(false)
if (Reference_global_vars.value_bool(Reference_global_vars.var_bool) != false)
    print "Runtime error test 2 \n"
    exit 1
end

Reference_global_vars.var_char = Reference_global_vars.createref_char('w')
if (Reference_global_vars.value_char(Reference_global_vars.var_char) != 'w')
    print "Runtime error test 3 \n"
    exit 1
end

Reference_global_vars.var_unsigned_char = Reference_global_vars.createref_unsigned_char(10)
if (Reference_global_vars.value_unsigned_char(Reference_global_vars.var_unsigned_char) != 10)
    print "Runtime error test 4 \n"
    exit 1
end

Reference_global_vars.var_signed_char = Reference_global_vars.createref_signed_char(10)
if (Reference_global_vars.value_signed_char(Reference_global_vars.var_signed_char) != 10)
    print "Runtime error test 5 \n"
    exit 1
end

Reference_global_vars.var_short = Reference_global_vars.createref_short(10)
if (Reference_global_vars.value_short(Reference_global_vars.var_short) != 10)
    print "Runtime error test 6 \n"
    exit 1
end

Reference_global_vars.var_unsigned_short = Reference_global_vars.createref_unsigned_short(10)
if (Reference_global_vars.value_unsigned_short(Reference_global_vars.var_unsigned_short) != 10)
    print "Runtime error test 7 \n"
    exit 1
end

Reference_global_vars.var_int = Reference_global_vars.createref_int(10)
if (Reference_global_vars.value_int(Reference_global_vars.var_int) != 10)
    print "Runtime error test 8 \n"
    exit 1
end

Reference_global_vars.var_unsigned_int = Reference_global_vars.createref_unsigned_int(10)
if (Reference_global_vars.value_unsigned_int(Reference_global_vars.var_unsigned_int) != 10)
    print "Runtime error test 9 \n"
    exit 1
end

Reference_global_vars.var_long = Reference_global_vars.createref_long(10)
if (Reference_global_vars.value_long(Reference_global_vars.var_long) != 10)
    print "Runtime error test 10 \n"
    exit 1
end

Reference_global_vars.var_unsigned_long = Reference_global_vars.createref_unsigned_long(10)
if (Reference_global_vars.value_unsigned_long(Reference_global_vars.var_unsigned_long) != 10)
    print "Runtime error test 11 \n"
    exit 1
end

Reference_global_vars.var_long_long = Reference_global_vars.createref_long_long(10)
if (Reference_global_vars.value_long_long(Reference_global_vars.var_long_long) != 10)
    print "Runtime error test 12 \n"
    exit 1
end

Reference_global_vars.var_unsigned_long_long = Reference_global_vars.createref_unsigned_long_long(10)
if (Reference_global_vars.value_unsigned_long_long(Reference_global_vars.var_unsigned_long_long) != 10)
    print "Runtime error test 13 \n"
    exit 1
end

Reference_global_vars.var_float = Reference_global_vars.createref_float(10.5)
if (Reference_global_vars.value_float(Reference_global_vars.var_float) != 10.5)
    print "Runtime error test 14 \n"
    exit 1
end

Reference_global_vars.var_double = Reference_global_vars.createref_double(10.5)
if (Reference_global_vars.value_double(Reference_global_vars.var_double) != 10.5)
    print "Runtime error test 15 \n"
    exit 1
end

