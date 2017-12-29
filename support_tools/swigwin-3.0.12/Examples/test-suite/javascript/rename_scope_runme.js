var rename_scope = require("rename_scope");

var a = new rename_scope.Natural_UP();
var b = new rename_scope.Natural_BP();

if (a.rtest() !== 1) {
  throw new Error("a.rtest(): Expected 1, was " + a.rtest());
}

if (b.rtest() !== 1) {
  throw new Error("b.rtest(): Expected 1, was " + b.rtest());
}

var f = rename_scope.equals;
if (f === undefined) {
  throw new Error("Equality operator has not been renamed.");
}
