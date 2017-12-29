var constructor_copy = require("constructor_copy");

f1 = new constructor_copy.Foo1(3);
f11 = new constructor_copy.Foo1(f1);

if (f1.x != f11.x) {
  throw "error in ctor copy for Foo1";
}

var good = 0;

f8 = new constructor_copy.Foo8()
try {
  f81 = new constructor_copy.Foo8(f8);
  good = 0;
} catch (err) {
  good = 1;
}

if (good == 0) {
  throw "Error: should not allow calling copy ctor for Foo8";
}


bi = new constructor_copy.Bari(5);
bc = new constructor_copy.Bari(bi);

if (bi.x != bc.x) {
  throw "Error in copy ctor of Bari";
}

bd = new constructor_copy.Bard(5);
try {
  bc = new constructor_copy.Bard(bd);
  good = 0;
} catch (err) {
  good = 1;
}

if (good == 0) {
  throw "Error: should not allow calling copy ctor for Bard";
}
