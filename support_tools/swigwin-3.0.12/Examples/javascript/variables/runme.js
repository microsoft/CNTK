var example = require("example");

// Try to set the values of some global variables
example.ivar   =  42;
example.svar   = -31000;
example.lvar   =  65537;
example.uivar  =  123456;
example.usvar  =  61000;
example.ulvar  =  654321;
example.scvar  =  -13;
example.ucvar  =  251;
example.cvar   =  "S";
example.fvar   =  3.14159;
example.dvar   =  2.1828;
example.strvar =  "Hello World";
example.iptrvar= example.new_int(37);
example.ptptr  = example.new_Point(37,42);
example.name   = "Bill";

// Now console.log out the values of the variables
console.log("Variables (values printed from Javascript)");
console.log("ivar      = " + example.ivar);
console.log("svar      = " + example.svar);
console.log("lvar      = " + example.lvar);
console.log("uivar     = " + example.uivar);
console.log("usvar     = " + example.usvar);
console.log("ulvar     = " + example.ulvar);
console.log("scvar     = " + example.scvar);
console.log("ucvar     = " + example.ucvar);
console.log("fvar      = " + example.fvar);
console.log("dvar      = " + example.dvar);
console.log("cvar      = " + example.cvar);
console.log("strvar    = " + example.strvar);
console.log("cstrvar   = " + example.cstrvar);
console.log("iptrvar   = " + example.iptrvar);
console.log("name      = " + example.name);
console.log("ptptr     = " + example.ptptr + ": " + example.Point_print(example.ptptr));
console.log("pt        = " + example.pt + ": " + example.Point_print(example.pt));


console.log("\nVariables (values printed from C)");

example.print_vars();

console.log("\nNow I'm going to try and modify some read only variables");

console.log("Tring to set 'path'");
try{
    example.path = "Whoa!";
    console.log("Hey, what's going on?!?! This shouldn't work");
}
catch(e){
    console.log("Good.");
}

console.log("Trying to set 'status'");
try{
    example.status = 0;
    console.log("Hey, what's going on?!?! This shouldn't work");
} catch(e){
    console.log("Good.");
}

console.log("\nI'm going to try and update a structure variable.");
example.pt = example.ptptr;
console.log("The new value is: ");
example.pt_print();
console.log("You should see the value: " + example.Point_print(example.ptptr));
