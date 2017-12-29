var class_ignore = require("class_ignore");

a = new class_ignore.Bar();

if (class_ignore.do_blah(a) != "Bar::blah")
    throw "Error";
