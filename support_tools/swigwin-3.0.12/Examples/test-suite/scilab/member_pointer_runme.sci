exec("swigtest.start", -1);

s = new_Square(10);

// Functions
checkequal(do_op(s, areapt()), 100.0, "Square area");
checkequal(do_op(s, perimeterpt()), 40.0, "Square perimeter");

// Variables
checkequal(do_op(s, areavar_get()), 100.0, "Square area");
areavar_set(perimeterpt());
checkequal(do_op(s, areavar_get()), 40.0, "Square perimeter");

// Constants
checkequal(do_op(s, AREAPT_get()), 100.0, "Square area");
checkequal(do_op(s, PERIMPT_get()), 40.0, "Square perimeter");

delete_Square(s);

exec("swigtest.quit", -1);
