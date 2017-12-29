var enum_template = require("enum_template");

if (enum_template.MakeETest() != 1)
  throw "RuntimeError";

if (enum_template.TakeETest(0) != null)
  throw "RuntimeError";

