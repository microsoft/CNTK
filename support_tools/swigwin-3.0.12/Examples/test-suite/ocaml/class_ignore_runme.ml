open Swig
open Class_ignore

let a = new_Bar C_void
let _ = (if _do_blah a <> C_string "Bar::blah" then
  raise (Failure "We didn't really get a bar object."))

