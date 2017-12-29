(* example_prog.ml *)

open Swig
open Example

(* Call our gcd() function *)

exception NoReturn

let x = 42 to int
let y = 105 to int
let g = _gcd '(x,y) as int
let _ = Printf.printf "The gcd of %d and %d is %d\n" (x as int) (y as int) g

(* Manipulate the Foo global variable *)

(* Output its current value *)
let _ = Printf.printf "Foo = %f\n" (_Foo '() as float)

(* Change its value *)
let _ = _Foo '(3.1415926)

(* See if the change took effect *)
let _ = Printf.printf "Foo = %f\n" (_Foo '() as float)









