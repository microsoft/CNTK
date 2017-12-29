octave_dim

assert(all(size(Foo45a())==[4 5]));
assert(all(size(Foo456a())==[4 5 6]));
assert(all(size(Foo4a())==[4 1]));
assert(all(size(Foo4b())==[4 1]));
assert(all(size(Foo())==[1 1]));
assert(all(size(Bar1())==[1 1]));
assert(all(size(Bar2())==[1 1]));
assert(all(size(Baz1())==[3 4]));
assert(all(size(Baz2())==[3 4]));
assert(all(size(Baz3())==[3 4]));
assert(all(size(Baz4())==[3 4]));

% Assertions will not work, but kept for future reference.
%assert(all(size(Baz5())==[3 4]));
%assert(all(size(Baz6())==[3 4]));
%assert(all(size(Baz7())==[3 4]));
