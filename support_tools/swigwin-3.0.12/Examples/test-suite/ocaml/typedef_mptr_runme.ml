open Swig
open Typedef_mptr

let soci x = (string_of_int (get_int x))

let x = new_Foo C_void 
let add_res = _do_op (C_list [ x ; C_int 2 ; C_int 1 ; _add ])
and sub_res = _do_op (C_list [ x ; C_int 2 ; C_int 1 ; _sub ])
let _ =
  if add_res <> (C_int 3) || sub_res <> (C_int 1) then
    raise (Failure ("Bad result:" ^
		    " (add " ^ (soci add_res) ^ ") " ^
		    " (sub " ^ (soci sub_res) ^ ")"))
let _ = Printf.printf "2 + 1 = %d, 2 - 1 = %d\n" 
	  (get_int add_res)
	  (get_int sub_res)
