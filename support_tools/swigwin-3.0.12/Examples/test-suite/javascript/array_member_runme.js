var array_member = require("array_member");

var f = new array_member.Foo();
f.data = array_member.global_data;

for (var i=0; i<8; i++) {
  if (array_member.get_value(f.data,i) != array_member.get_value(array_member.global_data,i)) {
    throw "Bad array assignment (1)";
  }
}

for (var i=0; i<8; i++) {
  array_member.set_value(f.data,i,-i);
}

array_member.global_data = f.data;

for (var i=0; i<8; i++){
  if (array_member.get_value(f.data,i) != array_member.get_value(array_member.global_data,i)) {
    throw "Bad array assignment (2)";
  }
}
