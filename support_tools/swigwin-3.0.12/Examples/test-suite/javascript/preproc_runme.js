var preproc = require("preproc");

if (preproc.endif != 1)
  throw "RuntimeError";

if (preproc.define != 1)
  throw "RuntimeError";

if (preproc.defined != 1)
  throw "RuntimeError";

if (2*preproc.one != preproc.two)
  throw "RuntimeError";

