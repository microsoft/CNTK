var example = require("example");

console.log("ICONST  = " + example.ICONST  + " (should be 42)");
console.log("FCONST  = " + example.FCONST  + " (should be 2.1828)");
console.log("CCONST  = " + example.CCONST  + " (should be 'x')");  
console.log("CCONST2 = " + example.CCONST2 + " (this should be on a new line)"); 
console.log("SCONST  = " + example.SCONST  + " (should be 'Hello World')");
console.log("SCONST2 = " + example.SCONST2 + " (should be '\"Hello World\"')");
console.log("EXPR    = " + example.EXPR    + " (should be 48.5484)");
console.log("iconst  = " + example.iconst  + " (should be 37)"); 
console.log("fconst  = " + example.fconst  + " (should be 3.14)"); 

console.log("EXTERN = " + example.EXTERN   + " (should be undefined)");
console.log("FOO    = " + example.FOO      + " (should be undefined)");
