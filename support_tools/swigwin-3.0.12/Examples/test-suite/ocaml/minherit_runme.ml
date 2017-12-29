(* Stolen from the python tests *)
open Swig
open Minherit

let a = new_Foo C_void
let b = new_Bar C_void 
let c = new_FooBar C_void
let d = new_Spam C_void

let soci x = (string_of_int (get_int x))

let _ =
  if (invoke a) "xget" C_void <> (C_int 1) then
    raise (Failure "Bad attribute value (a.xget)")

let _ =
  if (invoke b) "yget" C_void <> (C_int 2) then
    raise (Failure "Bad attribute value (b.yget)")

let _ =
  if   (invoke c) "xget" C_void <> (C_int 1)
    || (invoke c) "yget" C_void <> (C_int 2)
    || (invoke c) "zget" C_void <> (C_int 3) then
      raise (Failure "Bad attribute value c")
	
let _ =
  if   (invoke d) "xget" C_void <> (C_int 1)
    || (invoke d) "yget" C_void <> (C_int 2)
    || (invoke d) "zget" C_void <> (C_int 3)
    || (invoke d) "wget" C_void <> (C_int 4) then
      raise (Failure "Bad attribute value d")

let xga = _xget a
let _ =
  if xga <> (C_int 1) then
    raise (Failure ("Bad attribute value (xget a): " ^ (soci xga)))

let ygb = _yget b
let _ =
  if ygb <> (C_int 2) then
    raise (Failure ("Bad attribute value (yget b): " ^
		    (string_of_int (get_int ygb))))

let xgc = _xget c and ygc = _yget c and zgc = _zget c
let _ =
  if xgc <> (C_int 1) || ygc <> (C_int 2) || zgc <> (C_int 3) then
    raise (Failure ("Bad attribute value (xgc=" ^ (soci xgc) ^
		    " (sb 1) ygc=" ^ (soci ygc) ^
		    " (sb 2) zgc=" ^ (soci zgc) ^
		    " (sb 3))"))

let xgd = _xget d and ygd = _yget d and zgd = _zget d and wgd = _wget d
let _ =
  if   xgd <> (C_int 1) || ygd <> (C_int 2) 
    || zgd <> (C_int 3) || wgd <> (C_int 4) then
      raise (Failure ("Bad attribute value (xgd=" ^ (soci xgd) ^
		      " (sb 1) ygd=" ^ (soci ygd) ^
		      " (sb 2) zgd=" ^ (soci zgd) ^
		      " (sb 3)"))

(* Cleanse all of the pointers and see what happens *)

let aa = _toFooPtr a
let bb = _toBarPtr b
let cc = _toFooBarPtr c
let dd = _toSpamPtr d

let xgaa = (invoke aa) "xget" C_void
let _ =
  if xgaa <> (C_int 1) then
    raise (Failure ("Bad attribute value xgaa " ^ (soci xgaa)))
      
let ygbb = (invoke bb) "yget" C_void
let _ =
  if ygbb <> (C_int 2) then
    raise (Failure ("Bad attribute value ygbb " ^ (soci ygbb)))
      
let xgcc = (invoke cc) "xget" C_void
and ygcc = (invoke cc) "yget" C_void
and zgcc = (invoke cc) "zget" C_void
	     
let _ =
  if xgcc <> (C_int 1) || ygcc <> (C_int 2) || zgcc <> (C_int 3) then
    raise (Failure ("Bad attribute value (" ^
		    (soci xgcc) ^ " (sb 1) " ^
		    (soci ygcc) ^ " (sb 2) " ^
		    (soci zgcc) ^ " (sb 3))"))

let xgdd = (invoke dd) "xget" C_void
and ygdd = (invoke dd) "yget" C_void
and zgdd = (invoke dd) "zget" C_void
and wgdd = (invoke dd) "wget" C_void

let _ =
  if   xgdd <> (C_int 1) || ygdd <> (C_int 2)
    || zgdd <> (C_int 3) || wgdd <> (C_int 4) then
      raise (Failure ("Bad value: " ^
		      "xgdd=" ^ (soci xgdd) ^
		      "ygdd=" ^ (soci ygdd) ^
		      "zgdd=" ^ (soci zgdd) ^
		      "wgdd=" ^ (soci wgdd)))

let xgaa = _xget aa
and ygbb = _yget bb
and xgcc = _xget cc
and ygcc = _yget cc
and zgcc = _zget cc
and xgdd = _xget dd
and ygdd = _yget dd
and zgdd = _zget dd
and wgdd = _wget dd

let _ = 
  if xgaa <> (C_int 1) then
    raise (Failure ("Fn xget: xgaa=" ^ (soci xgaa)))

let _ =
  if ygbb <> (C_int 2) then
    raise (Failure ("Fn yget: ygbb=" ^ (soci ygbb)))

let _ =
  if   xgcc <> (C_int 1) || ygcc <> (C_int 2) || zgcc <> (C_int 3) then
    raise (Failure ("CC with fns: (" ^ 
		    (soci xgcc) ^ " " ^ (soci ygcc) ^ " " ^ (soci zgcc)))

let _ =
  if   xgdd <> (C_int 1) || ygdd <> (C_int 2) 
    || zgdd <> (C_int 3) || wgdd <> (C_int 4) then
    raise (Failure ("CC with fns: (" ^ 
		    (soci xgdd) ^ " " ^ (soci ygdd) ^ " " ^ 
		    (soci zgdd) ^ " " ^ (soci wgdd) ^ ")"))
