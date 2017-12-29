var abstract_inherit = require("abstract_inherit");

// Shouldn't be able to instantiate any of these classes
// since none of them implements the pure virtual function
// declared in the base class (Foo).
var Foo = abstract_inherit.Foo;
var Bar = abstract_inherit.Bar;
var Spam = abstract_inherit.Spam;

var caughtException = false;
try {
  new Foo();
} catch (err) {
  caughtException = true;
}
if (!caughtException) {
  throw new Error("Foo should be instantiated as it is abstract");
}

caughtException = false;
try {
  new Bar();
} catch (err) {
  caughtException = true;
}

if (!caughtException) {
  throw new Error("Bar should be instantiated as it is abstract");
}

caughtException = false;
try {
  new Spam();  
} catch (err) {
  caughtException = true;
}

if (!caughtException) {
  throw new Error("Spam should be instantiated as it is abstract");
}
