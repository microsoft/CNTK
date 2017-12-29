# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

li_std_vector_enum

function check(a, b)
  if (a != b)
    error("incorrect match");
  endif
end

ev = EnumVector();

check(ev.nums(0), 10);
check(ev.nums(1), 20);
check(ev.nums(2), 30);

it = ev.nums.begin();
v = it.value();
check(v, 10);
it.next();
v = it.value();
check(v, 20);

#expected = 10 
#ev.nums.each do|val|
#  swig_assert(val == expected)
#  expected += 10
#end

