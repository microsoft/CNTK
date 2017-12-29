var example = require("example");

// ----- Object creation -----

// Print out the value of some enums
console.log("*** color ***");
console.log("    RED    =" + example.RED);
console.log("    BLUE   =" + example.BLUE);
console.log("    GREEN  =" + example.GREEN);

console.log("\n*** Foo::speed ***");
console.log("    Foo_IMPULSE   =" + example.Foo.IMPULSE);
console.log("    Foo_WARP      =" + example.Foo.WARP);
console.log("    Foo_LUDICROUS =" + example.Foo.LUDICROUS);

console.log("\nTesting use of enums with functions\n");

example.enum_test(example.RED, example.Foo.IMPULSE);
example.enum_test(example.BLUE,  example.Foo.WARP);
example.enum_test(example.GREEN, example.Foo.LUDICROUS);
example.enum_test(1234,5678);

console.log("\nTesting use of enum with class method");
f = new example.Foo();

f.enum_test(example.Foo.IMPULSE);
f.enum_test(example.Foo.WARP);
f.enum_test(example.Foo.LUDICROUS);

// enum value BLUE of enum color is accessed as property of cconst
console.log("example.BLUE= " + example.BLUE);

// enum value LUDICROUS of enum Foo::speed is accessed as as property of cconst
console.log("example.speed.LUDICROUS= " + example.Foo.LUDICROUS); 
