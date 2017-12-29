var cpp_enum = require("cpp_enum");

var f = new cpp_enum.Foo()

if(f.hola != cpp_enum.Hello){
  print(f.hola);
  throw "Error";
}

f.hola = cpp_enum.Foo.Hi
if(f.hola != cpp_enum.Foo.Hi){
  print(f.hola);
  throw "Error";
}

f.hola = cpp_enum.Hello

if(f.hola != cpp_enum.Hello){
  print(f.hola);
  throw "Error";
}

cpp_enum.Foo.hi = cpp_enum.Hello
if(cpp_enum.Foo.hi != cpp_enum.Hello){
  print(cpp_enum.Foo.hi);
  throw "Error";
}

