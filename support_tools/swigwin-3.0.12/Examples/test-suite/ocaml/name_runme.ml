open Swig
open Name 

let _ = if (get_int (_Baz_2 C_void)) - (get_int (_bar_2 C_void)) == 30 
then 
  exit 0 
else 
  exit 1
  
