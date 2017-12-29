var example = require("example");

// ----- Object creation -----

console.log("Creating some objects:");
c = new example.Circle(10);
console.log("Created circle " + c);
s = new example.Square(10);
console.log("Created square " + s);

// ----- Access a static member -----
console.log("\nA total of " + example.Shape.nshapes + " shapes were created"); // access static member as properties of the class object

// ----- Member data access -----
// Set the location of the object.
// Note: methods in the base class Shape are used since
// x and y are defined there.

c.x = 20;
c.y = 30;
s.x = -10;
s.y = 5;

console.log("\nHere is their new position:");
console.log("Circle = (" + c.x + "," + c.y + ")");
console.log("Square = (" + s.x + "," + s.y + ")");

// ----- Call some methods -----
console.log("\nHere are some properties of the shapes:");
console.log("Circle:");
console.log("area = " + c.area() + "");
console.log("perimeter = " + c.perimeter() + "");
console.log("\n");
console.log("Square:");
console.log("area = " + s.area() + "");
console.log("perimeter = " + s.perimeter() + "");

// ----- Delete everything -----
console.log("\nGuess I'll clean up now");
// Note: this invokes the virtual destructor
delete c;
delete s;

console.log(example.Shape.nshapes + " shapes remain");

console.log("Goodbye");
