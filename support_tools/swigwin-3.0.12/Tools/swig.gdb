# User-defined commands for easier debugging of SWIG in gdb
#
# This file can be "included" into your main .gdbinit file using:
# source swig.gdb
# or otherwise paste the contents into .gdbinit
#
# Note all user defined commands can be seen using:
# (gdb) show user
# The documentation for each command can be easily viewed, for example:
# (gdb) help swigprint

define swigprint
    if ($argc == 2)
        set $expand_count = $arg1
    else
        set $expand_count = -1
    end
    Printf "%s\n", Swig_to_string($arg0, $expand_count)
end
document swigprint
Displays any SWIG DOH object
Usage: swigprint swigobject [hashexpandcount]
  swigobject      - The object to display.
  hashexpandcount - Number of nested Hash types to expand (default is 1). See Swig_set_max_hash_expand() to change default.
end


define locswigprint
    if ($argc == 2)
        set $expand_count = $arg1
    else
        set $expand_count = -1
    end
    Printf "%s\n", Swig_to_string_with_location($arg0, $expand_count)
end
document locswigprint
Displays any SWIG DOH object prefixed with file and line location
Usage: locswigprint swigobject [hashexpandcount]
  swigobject      - The object to display.
  hashexpandcount - Number of nested Hash types to expand (default is 1). See Swig_set_max_hash_expand() to change default.
end
