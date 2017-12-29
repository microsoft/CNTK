var null_pointer = require("null_pointer");

if (!null_pointer.func(null)) {
  throw new Error("Javascript 'null' should be converted into NULL.");
}

if (null_pointer.getnull() != null) {
  throw new Error("NULL should be converted into Javascript 'null'.");
}
