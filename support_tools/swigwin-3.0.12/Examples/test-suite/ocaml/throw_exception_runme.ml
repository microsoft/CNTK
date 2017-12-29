(* Throw exception test *)

open Swig
open Throw_exception

let x = new_Foo C_void ;;
let _ =
  try
    (invoke x) "test_int" C_void 
  with (Failure "Exception(37): Thrown exception from C++ (int)\n") ->
  try 
    (invoke x) "test_msg" C_void
  with (Failure "Exception(0): Dead\n") ->
  try
    (invoke x) "test_cls" C_void 
  with (Failure "Exception(0): Thrown exception from C++ (unknown)\n") ->
  try
    (invoke x) "test_multi" (C_int 1)
  with (Failure "Exception(37): Thrown exception from C++ (int)\n") ->
  try
    (invoke x) "test_multi" (C_int 2)
  with (Failure "Exception(0): Dead\n") ->
  try
    (invoke x) "test_multi" (C_int 3)
  with (Failure "Exception(0): Thrown exception from C++ (unknown)\n") ->
    exit 0

let _ = exit 1
