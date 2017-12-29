var nspace_extend = require("nspace_extend");

// constructors and destructors
var color1 = new nspace_extend.Outer.Inner1.Color();
var color = new nspace_extend.Outer.Inner1.Color(color1);
delete color1;

// class methods
color.colorInstanceMethod(20.0);
nspace_extend.Outer.Inner1.Color.colorStaticMethod(20.0);
var created = nspace_extend.Outer.Inner1.Color.create();


// constructors and destructors
var color2 = new nspace_extend.Outer.Inner2.Color();
color = new nspace_extend.Outer.Inner2.Color(color2);
delete color2;

// class methods
color.colorInstanceMethod(20.0);
nspace_extend.Outer.Inner2.Color.colorStaticMethod(20.0);
created = nspace_extend.Outer.Inner2.Color.create();

// Same class different namespaces
var col1 = new nspace_extend.Outer.Inner1.Color();
var col2 = nspace_extend.Outer.Inner2.Color.create();
col2.colors(col1, col1, col2, col2, col2);
