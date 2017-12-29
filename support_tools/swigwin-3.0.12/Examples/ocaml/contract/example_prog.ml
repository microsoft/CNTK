open Swig 
open Example

let _ = print_endline "This won't throw." 
let _ = Printf.printf "Cos 1.0 is %f\n" (_cos '(1.0) as float)
let _ = print_endline "This will throw."
let _ = Printf.printf "Cos 5.0 is %f\n" (_cos '(5.0) as float)
