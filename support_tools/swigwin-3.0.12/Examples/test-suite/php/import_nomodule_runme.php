<?php
require "tests.php";
require "import_nomodule.php";

// No new functions
check::functions(array(create_foo,delete_foo,test1,is_python_builtin));
// No new classes
check::classes(array(import_nomodule,Bar));
// now new vars
check::globals(array());

$f = import_nomodule::create_Foo();
import_nomodule::test1($f,42);
import_nomodule::delete_Foo($f);

$b = new Bar();
import_nomodule::test1($b,37);

check::done();
?>
