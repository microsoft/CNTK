var char_binary = require("char_binary");

var t = new char_binary.Test();
if (t.strlen('hile') != 4) {
  print(t.strlen('hile'));
  throw("bad multi-arg typemap 1");
}
if (t.ustrlen('hile') != 4) {
  print(t.ustrlen('hile'));
  throw("bad multi-arg typemap 1");
}

if (t.strlen('hil\0') != 4) {
  throw("bad multi-arg typemap 2");
}
if (t.ustrlen('hil\0') != 4) {
  throw("bad multi-arg typemap 2");
}

/*
 *  creating a raw char*
 */
var pc = char_binary.new_pchar(5);
char_binary.pchar_setitem(pc, 0, 'h');
char_binary.pchar_setitem(pc, 1, 'o');
char_binary.pchar_setitem(pc, 2, 'l');
char_binary.pchar_setitem(pc, 3, 'a');
char_binary.pchar_setitem(pc, 4, 0);


if (t.strlen(pc) != 4) {
  throw("bad multi-arg typemap (3)");
}
if (t.ustrlen(pc) != 4) {
  throw("bad multi-arg typemap (3)");
}

char_binary.var_pchar = pc;
if (char_binary.var_pchar != "hola") {
  print(char_binary.var_pchar);
  throw("bad pointer case (1)");
}

char_binary.var_namet = pc;
if (char_binary.var_namet != "hola") {
  throw("bad pointer case (2)");
}
char_binary.delete_pchar(pc);
