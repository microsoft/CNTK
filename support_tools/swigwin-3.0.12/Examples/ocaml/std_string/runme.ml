(* This example was mostly lifted from the guile example directory *)

open Swig
open Example

let y = "\205\177"
let z = _to_wstring_with_locale '((y to string),(Sys.argv.(1) to string))

let _ = 
  begin
    print_string "the original string contains " ;
    print_int (String.length y) ;
    print_newline () ;
    
    print_string "the new string contains " ;
    print_int (z -> size () as int) ;
    print_string " : [ " ;
    for i = 0 to (pred ((z -> size ()) as int)) do
      print_int ((z '[i to int]) as int) ;
      print_string "; " ;
    done ;
    print_string "]" ;
    print_newline () ;
  end    
