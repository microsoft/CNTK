(* This example was mostly lifted from the guile example directory *)

open Swig
open Example

let with_vector v f =
  for i = 0 to ((v -> size()) as int) - 1 do
    f v i
  done

let print_DoubleVector v =
  begin
    with_vector v 
      (fun v i -> 
	 print_float ((v '[i to int]) as float) ;
	 print_string " ") ;
    print_endline 
  end

(* Call average with a Ocaml array... *)

let v = new_DoubleVector '()
let rec fill_dv v x =
  if x < 0.0001 then v else 
    begin
      v -> push_back ((x to float)) ;
      fill_dv v (x *. x)
    end
let _ = fill_dv v 0.999
let _ = print_DoubleVector v ; print_endline ""
let u = new_IntVector '()
let _ = for i = 1 to 4 do
  u -> push_back ((i to int))
done
let _ = (print_float ((_average u) as float) ; print_newline ())
