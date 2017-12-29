# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

# load module
clear all;
swigexample;
assert(cvar.ivar == ifunc);
assert(exist("swigexample","var"));
clear all
swigexample;
assert(cvar.ivar == ifunc);
assert(exist("swigexample","var"));
clear all

# load module in a function globally before base context
clear all;
function testme
  swigexample;
  assert(cvar.ivar == ifunc);
  assert(exist("swigexample","var"));
endfunction
testme
testme
swigexample;
assert(cvar.ivar == ifunc);
assert(exist("swigexample","var"));
clear all
function testme
  swigexample;
  assert(cvar.ivar == ifunc);
  assert(exist("swigexample","var"));
endfunction
testme
testme
swigexample;
assert(cvar.ivar == ifunc);
assert(exist("swigexample","var"));
clear all

# load module in a function globally after base context
clear all;
swigexample;
assert(cvar.ivar == ifunc);
assert(exist("swigexample","var"));
function testme
  swigexample;
  assert(cvar.ivar == ifunc);
  assert(exist("swigexample","var"));
endfunction
testme
testme
clear all
swigexample;
assert(cvar.ivar == ifunc);
assert(exist("swigexample","var"));
function testme
  swigexample;
  assert(cvar.ivar == ifunc);
  assert(exist("swigexample","var"));
endfunction
testme
testme
clear all

# octave 3.0.5 randomly crashes on the remaining tests
if !swig_octave_prereq(3,2,0)
  exit
endif

# load module with no cvar
clear all;
swigexample2;
assert(swigexample2.ivar == ifunc);
assert(exist("swigexample2","var"));
assert(!isglobal("cvar"))
clear all
swigexample2;
assert(swigexample2.ivar == ifunc);
assert(exist("swigexample2","var"));
assert(!isglobal("cvar"))
clear all
