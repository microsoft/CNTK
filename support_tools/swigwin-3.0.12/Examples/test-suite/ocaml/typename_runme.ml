(* Fun with type names -- stolen from the ruby runme *)

open Swig
open Typename

let f = new_Foo C_void 
let b = new_Bar C_void

let x = _twoFoo f 
let _ = match x with C_double f -> () | _ -> raise (Failure "not a float")
let y = _twoBar b
let _ = match y with C_int i -> () | _ -> raise (Failure "not an int")
