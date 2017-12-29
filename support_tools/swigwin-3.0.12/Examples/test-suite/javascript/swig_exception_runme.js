var swig_exception = require("swig_exception");

var c = new swig_exception.Circle(10);
var s = new swig_exception.Square(10);

if (swig_exception.Shape.nshapes != 2) {
    throw "Shape.nshapes should be 2, actually " + swig_exception.Shape.nshapes;
}

// ----- Throw exception -----
try {
    c.throwException();
    throw "Exception wasn't thrown";
} catch (e) {
    if (e.message != "OK") {
	throw "Exception message should be \"OK\", actually \"" + e.message + "\"";
    }
}

// ----- Delete everything -----

c = null;
s = null;
e = null;

/* FIXME: Garbage collection needs to happen before this check will work.
if (swig_exception.Shape.nshapes != 0) {
    throw "Shape.nshapes should be 0, actually " + swig_exception.Shape.nshapes;
}
*/
