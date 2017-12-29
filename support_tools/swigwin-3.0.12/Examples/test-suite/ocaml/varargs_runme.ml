(* Test case stolen from the python directory *)

open Swig
open Varargs

let _ = if _test (C_string "Hello") <> (C_string "Hello") then
    raise (Failure "1")

let f = new_Foo C_void
let _ = if (invoke f) "test" (C_string "Hello") <> (C_string "Hello") then
    raise (Failure "2")
