(* Test the unions example... *)

open Swig
open Unions 

let a = new_SmallStruct C_void
let b = new_BigStruct C_void
let c = new_UnionTest C_void
let d = new_EmbeddedUnionTest C_void 

let _ = (invoke a) "jill" (C_short 3)
let _ = (invoke b) "jack" (C_char 'a')  (* Int conversion *)
let _ = (invoke b) "smallstruct" a      (* Put a in b *)
let _ = (invoke c) "bs" b

let _ = if get_int ((invoke a) "jill" C_void) != 3 then
	raise (Failure "jill value is not preserved")
let _ = if get_int ((invoke b) "jack" C_void) != (int_of_char 'a') then
	raise (Failure "jack value is not preserved")
let _ = if get_int ((invoke ((invoke b) "smallstruct" C_void))
	"jill" C_void) != 3 then
	raise (Failure "jill value is not embedded in bigstruct")
let _ = if get_int ((invoke ((invoke c) "bs" C_void))
	"jack" C_void) != (int_of_char 'a') then
	raise (Failure "union set of bigstruct did not take")
let _ = if get_int ((invoke ((invoke c) "ss" C_void))
	"jill" C_void) != (int_of_char 'a') then
	raise (Failure "corresponding union values are not the same")
