(* This example is meant to reach every case in cstring.i *)

open Swig
open Example

let _ = _takes_std_string (C_string "foo")
let _ = print_endline 
  ("_gives_std_string <<" ^ (get_string (_gives_std_string C_void)) ^ " >>")
let _ = _takes_char_ptr (C_string "bar")
let _ = print_endline 
  ("_gives_char_ptr << " ^ (get_string (_gives_char_ptr C_void)) ^ " >>")
let _ = print_endline
  ("_takes_and_gives_std_string << " ^ 
   (get_string (_takes_and_gives_std_string (C_string "foo"))) ^ " >>")
let _ = print_endline
  ("_takes_and_gives_char_ptr << " ^
   (get_string (_takes_and_gives_char_ptr (C_string "bar.bar"))) ^ " >>")
