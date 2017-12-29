var example = require("example");

// First create some objects using the pointer library.
console.log("Testing the pointer library");
a = example.new_intp();
b = example.new_intp();
c = example.new_intp();

example.intp_assign(a,37);
example.intp_assign(b,42);

console.log(" a = " + example.intp_value(a));
console.log(" b = " + example.intp_value(b));
console.log(" c = " + example.intp_value(c));

//// Call the add() function with some pointers
example.add(a, b, c);

//
//// Now get the result
r = example.intp_value(c);
console.log(" 37 + 42 = " + r);

// Clean up the pointers
example.delete_intp(a);
example.delete_intp(b);
example.delete_intp(c);

//// Now try the typemap library
//// This should be much easier. Now how it is no longer
//// necessary to manufacture pointers.
//"OUTPUT" Mapping is not supported
//console.log("Trying the typemap library");
//r = example.subtract(37,42);
//console.log("37 - 42 =" + r);
