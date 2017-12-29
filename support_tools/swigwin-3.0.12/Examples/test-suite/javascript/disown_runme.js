var disown = require("disown");

var a = new disown.A();
var tmp = a.thisown;
a.thisown = 0
if (a.thisown) {
  throw new Error("Failed.");
}
a.thisown = 1
if (!a.thisown) {
  throw new Error("Failed.");
}
a.thisown = tmp
if (a.thisown != tmp) {
  throw new Error("Failed.");
}

var b = new disown.B();
b.acquire(a);
if (a.thisown) {
  throw new Error("Failed.");
}
