(* Stolen from ruby test cases *)

open Swig
open Sneaky1

let x = Sneaky1._add (C_list [ C_int 3; C_int 4 ])
let y = Sneaky1._subtract (C_list [ C_int 3; C_int 4 ])
let z = Sneaky1._mul (C_list [ C_int 3; C_int 4 ])
let w = Sneaky1._divide (C_list [ C_int 3; C_int 4 ])
