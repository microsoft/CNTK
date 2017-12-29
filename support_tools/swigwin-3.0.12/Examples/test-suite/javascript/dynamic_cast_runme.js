var dynamic_cast = require("dynamic_cast");

var f = new dynamic_cast.Foo();
var b = new dynamic_cast.Bar();

var x = f.blah();
var y = b.blah();

var a = dynamic_cast.do_test(y);
if (a != "Bar::test") {
  throw new Error("Failed.");
}
