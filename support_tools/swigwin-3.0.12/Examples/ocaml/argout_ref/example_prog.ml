(* example_prog.ml *)

open Swig
open Example

exception BadReturn

let _ = if Array.length Sys.argv < 3 then
  begin
    print_endline 
      ("Usage: " ^ Sys.argv.(0) ^ " n1 n2\n" ^
       " Displays the least factors of the numbers that have the same\n" ^
       " relationship, 16 12 -> 4 3\n") ;
    exit 0
  end

let x = int_of_string Sys.argv.(1)
let y = int_of_string Sys.argv.(2)
let (xf,yf) = match _factor '((x to int),(y to int)) with
    C_list [ C_int a ; C_int b ] -> a,b
  | _ -> raise BadReturn
let _ = print_endline
	  ("Factorization of " ^ (string_of_int x) ^ 
	   " and " ^ (string_of_int y) ^ 
	   " is " ^ (string_of_int xf) ^ 
	   " and " ^ (string_of_int yf))
