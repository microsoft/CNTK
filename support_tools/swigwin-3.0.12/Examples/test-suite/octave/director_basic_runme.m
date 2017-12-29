director_basic


function self=OctFoo()
  global director_basic;
  self=subclass(director_basic.Foo());
  self.ping=@OctFoo_ping;
end
function string=OctFoo_ping(self)
  string="OctFoo::ping()";
end

a = OctFoo();

if (!strcmp(a.ping(),"OctFoo::ping()"))
  error(a.ping())
endif

if (!strcmp(a.pong(),"Foo::pong();OctFoo::ping()"))
  error(a.pong())
endif

b = director_basic.Foo();

if (!strcmp(b.ping(),"Foo::ping()"))
  error(b.ping())
endif

if (!strcmp(b.pong(),"Foo::pong();Foo::ping()"))
  error(b.pong())
endif

a = director_basic.A1(1);

if (a.rg(2) != 2)
  error
endif

function self=OctClass()
  global director_basic;
  self=subclass(director_basic.MyClass());
  self.method=@OctClass_method;
  self.vmethod=@OctClass_vmethod;
end
function OctClass_method(self,vptr)
  self.cmethod = 7;
end
function out=OctClass_vmethod(self,b)
  b.x = b.x + 31;
  out=b;
end

b = director_basic.Bar(3);
d = director_basic.MyClass();
c = OctClass();

cc = director_basic.MyClass_get_self(c);
dd = director_basic.MyClass_get_self(d);

bc = cc.cmethod(b);
bd = dd.cmethod(b);

cc.method(b);
if (c.cmethod != 7)
  error
endif

if (bc.x != 34)
  error
endif


if (bd.x != 16)
  error
endif


function self=OctMulti()
  global director_basic;
  self=subclass(director_basic.Foo(),director_basic.MyClass());
  self.vmethod=@OctMulti_vmethod;
  self.ping=@OctMulti_ping;
end
function out=OctMulti_vmethod(self,b)
  b.x = b.x + 31;
  out=b;
end
function out=OctMulti_ping(self)
  out="OctFoo::ping()";
end

a = 0;
for i=0:100,
    octmult = OctMulti();
    octmult.pong();
    clear octmult 
endfor


octmult = OctMulti();


p1 = director_basic.Foo_get_self(octmult);
p2 = director_basic.MyClass_get_self(octmult);

p1.ping();
p2.vmethod(bc);



