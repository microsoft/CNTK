<?php
require "tests.php";
require "arrays.php";

check::functions(array(fn_taking_arrays,newintpointer,setintfrompointer,getintfrompointer,array_pointer_func));
check::classes(array(arrays,SimpleStruct,ArrayStruct,CartPoseData_t));
check::globals(array(simplestruct_double_field,arraystruct_array_c,arraystruct_array_sc,arraystruct_array_uc,arraystruct_array_s,arraystruct_array_us,arraystruct_array_i,arraystruct_array_ui,arraystruct_array_l,arraystruct_array_ul,arraystruct_array_ll,arraystruct_array_f,arraystruct_array_d,arraystruct_array_struct,arraystruct_array_structpointers,arraystruct_array_ipointers,arraystruct_array_enum,arraystruct_array_enumpointers,arraystruct_array_const_i,cartposedata_t_p));

$ss=new simplestruct();
check::classname(simplestruct,$ss);

$as=new arraystruct();
$as->array_c="abc";
check::equal($as->array_c,"a",'$as->array_c=="a"');
check::equal(isset($as->array_const_i),TRUE,'isset($as->array_const_i)');

check::done();
?>
