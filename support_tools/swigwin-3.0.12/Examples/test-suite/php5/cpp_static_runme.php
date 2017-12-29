<?php

require "tests.php";
require "cpp_static.php";

// New functions
check::functions(array(staticfunctiontest_static_func,staticfunctiontest_static_func_2,staticfunctiontest_static_func_3,is_python_builtin));
// New classes
check::classes(array(StaticMemberTest,StaticFunctionTest,cpp_static,StaticBase,StaticDerived));
// New vars
check::globals(array(staticmembertest_static_int,staticbase_statty,staticderived_statty));

check::done();
?>
