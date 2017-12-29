var complextest = require("complextest");

a = [-1,2];

expected = [-1, -2];

a_c = complextest.Conj(a);
if (a_c.toString() != expected.toString())
  throw "Error in Conj(a)";

a_c_f = complextest.Conjf(a);
if (a_c_f.toString() != expected.toString())
    throw "Error in Conjf(a)";

v = new complextest.VectorStdCplx();
v.add([1,2]);
v.add([2,3]);
v.add([4,3]);
v.add(1);

// TODO: how to check validity?
complextest.Copy_h(v);
