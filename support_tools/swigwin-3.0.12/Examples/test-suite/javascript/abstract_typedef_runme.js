var abstract_typedef = require("abstract_typedef");

var e = new abstract_typedef.Engine();
var a = new abstract_typedef.A()

if (a.write(e) != 1) {
  throw "Error";
}
