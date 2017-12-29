(* This example was mostly lifted from the guile example directory *)

open Swig
open Example

let v = new_StringVector '() 

let _ = 
  for i = 0 to (Array.length Sys.argv) - 1 do
    let str = (Sys.argv.(i)) to string in v -> push_back (str)
  done

let _ = _vec_write '(v)
