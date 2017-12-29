# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

director_stl

MyFoo=@() subclass(director_stl.Foo(),\
  'ping',@(self,s) "MyFoo::ping():" + s,\
  'pident',@(self,arg) arg,\
  'vident',@(self,v) v,\
  'vidents',@(self,v) v,\
  'vsecond',@(self,v1,v2) v2,\
);

a = MyFoo();

a.tping("hello");
a.tpong("hello");

p = {1,2}
a.pident(p);
v = {3,4}
a.vident(v);

a.tpident(p);
a.tvident(v);

v1 = {3,4};
v2 = {5,6};
a.tvsecond(v1,v2);

vs=("hi", "hello");
vs;
a.tvidents(vs);

