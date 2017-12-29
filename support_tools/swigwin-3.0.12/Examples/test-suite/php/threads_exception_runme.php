<?php

require "tests.php";
require "threads_exception.php";

// Check functions
check::functions(array(test_simple,test_message,test_hosed,test_unknown,test_multi,is_python_builtin));
// Check classes.
check::classes(array(Exc,Test,threads_exception));
// Chek globals.
check::globals(array(exc_code,exc_msg));

$t = new Test();
try {
    $t->unknown();
} catch (Exception $e) {
    check::equal($e->getMessage(), 'C++ A * exception thrown', '');
}

try {
    $t->simple();
} catch (Exception $e) {
    check::equal($e->getCode(), 37, '');
}

try {
    $t->message();
} catch (Exception $e) {
    check::equal($e->getMessage(), 'I died.', '');
}

try {
    $t->hosed();
} catch (Exception $e) {
    check::equal($e->getMessage(), 'C++ Exc exception thrown', '');
}

foreach (Array(1,2,3,4) as $i) {
    try {
	$t->multi($i);
    } catch (Exception $e) {
    }
}
