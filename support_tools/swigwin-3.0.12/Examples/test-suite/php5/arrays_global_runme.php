<?php

require "tests.php";
require "arrays_global.php";

check::functions(array(test_a,test_b,new_simplestruct,new_material));
check::classes(array(arrays_global,SimpleStruct,Material));
check::globals(array(array_c,array_sc,array_uc,array_s,array_us,array_i,array_ui,array_l,array_ul,array_ll,array_f,array_d,array_struct,array_structpointers,array_ipointers,array_enum,array_enumpointers,array_const_i,beginstring_fix44a,beginstring_fix44b,beginstring_fix44c,beginstring_fix44d,beginstring_fix44e,beginstring_fix44f,chitmat,hitmat_val,hitmat,simplestruct_double_field));
// The size of array_c is 2, but the last byte is \0, so we can only store a
// single byte string in it.
check::set(array_c,"Z");
check::equal("Z",check::get(array_c),"set array_c");
check::set(array_c,"xy");
check::equal("x",check::get(array_c),"set array_c");
check::set(array_c,"h");
check::equal("h",check::get(array_c),"set array_c");

check::done();
?>
