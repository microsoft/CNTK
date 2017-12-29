var example = require("example");

a = new example.Complex(2,3);
b = new example.Complex(-5,10);

console.log ("a =" + a);
console.log ("b   =" + b);

c = a.plus(b);

console.log("c   =" + c);
console.log("a*b =" + a.times(b));
console.log("a-c =" + a.minus(c));

e = example.Complex.copy(a.minus(c));
console.log("e   =" + e);

// Big expression
f = a.plus(b).times(c.plus(b.times(e))).plus(a.uminus());
console.log("f   =" + f);





