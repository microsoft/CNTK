(* example_prog.ml *)

open Swig ;;
open Example ;;

let side_length (ax,ay) (bx,by) =
  sqrt (((bx -. ax) ** 2.0) +. ((by -. ay) ** 2.0)) ;;

let triangle_area a_pt b_pt c_pt =
  let a = (side_length a_pt b_pt) 
  and b = (side_length b_pt c_pt)
  and c = (side_length c_pt a_pt) in
  let s = (a +. b +. c) /. 2.0 in
    sqrt (s *. (s -. a) *. (s -. b) *. (s -. c)) ;;

let point_in_triangle (pta,ptb,ptc) x y =
  let delta = 0.0000001 in (* Error *)
  let ptx = (x,y) in
    begin
      let a_area = triangle_area pta ptb ptx
      and b_area = triangle_area ptb ptc ptx
      and c_area = triangle_area ptc pta ptx
      and x_area = triangle_area pta ptb ptc in
      let result = (abs_float (a_area +. b_area +. c_area -. x_area)) < delta
      in
	result
    end ;;

let triangle_class pts ob meth args =
  match meth with
      "cover" ->
	(match args with
	     C_list [ x_arg ; y_arg ] ->
	       let xa = x_arg as float 
	       and ya = y_arg as float in
		 (point_in_triangle pts xa ya) to bool
	   | _ -> raise (Failure "cover needs two double arguments."))
    | _ -> (invoke ob) meth args ;;

let dist (ax,ay) (bx,by) = 
  let dx = ax -. bx and dy = ay -. by in
    sqrt ((dx *. dx) +. (dy *. dy))

let waveplot_depth events distance pt =
  (List.fold_left (+.) 0.0 
     (List.map 
	(fun (x,y,d) -> 
	   let t = dist pt (x,y) in
	     ((sin t) /. t) *. d)
	events)) +. distance

let waveplot_class events distance ob meth args =
  match meth with
      "depth" ->
	(match args with
	     C_list [ x_arg ; y_arg ] ->
	       let xa = x_arg as float 
	       and ya = y_arg as float in
		 (waveplot_depth events distance (xa,ya)) to float
	   | _ -> raise (Failure "cover needs two double arguments."))
    | _ -> (invoke ob) meth args ;;

let triangle =
  new_derived_object 
    new_shape
    (triangle_class ((0.0,0.0),(0.5,1.0),(1.0,0.6)))
    '() ;;

let waveplot = 
  new_derived_object
    new_volume
    (waveplot_class [ 0.01,0.01,3.0 ; 1.01,-2.01,1.5 ] 5.0)
    '() ;;

let _ = _draw_shape_coverage '(triangle, 60, 20) ;;
let _ = _draw_depth_map '(waveplot, 60, 20) ;;
