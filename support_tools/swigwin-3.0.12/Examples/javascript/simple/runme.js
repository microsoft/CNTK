var example = require("example");

/* Call our gcd() function */

x = 42;
y = 105;
g = example.gcd(x,y);
console.log("GCD of x and y is=" + g);

/* Manipulate the Foo global variable */

/* Output its current value */
console.log("Global variable Foo=" + example.Foo);

/* Change its value */
example.Foo = 3.1415926;

/* See if the change took effect */
console.log("Variable Foo changed to=" + example.Foo);




 
 
 
