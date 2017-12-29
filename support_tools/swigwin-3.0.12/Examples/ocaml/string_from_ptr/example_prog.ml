(* foo_program.ml -- the program using foolib *)

open Swig    (* Give access to the swig library *)
open Foolib  (* This is the name of your swig output *)

let results = _foo '()  (* Function names are prefixed with _ in order to make
			   them lex as identifiers in ocaml.  Consider that
			   uppercase identifiers are module names in ocaml.
			   NOTE: the '() syntax is part of swigp4.  You can do:
		           let results = _foo C_void *)

(* Since your function has a return value in addition to the string output,
   you'll need to match them as a list *)

let result_string =
  match results with 
    C_list [ C_string result_string ; C_int 0 ] -> (* The return value is
	last when out arguments appear, but this too can be customized.
	We're also checking that the function succeeded. *)
      result_string
  | _ -> raise (Failure "Expected string, int reply from _foo")
      
let _ = print_endline result_string
