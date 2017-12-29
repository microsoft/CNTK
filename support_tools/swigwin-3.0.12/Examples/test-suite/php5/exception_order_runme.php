<?
require "tests.php";
require "exception_order.php";

check::functions(array(a_foo,a_bar,a_foobar,a_barfoo,is_python_builtin));
check::classes(array(A,E1,E2,E3,exception_order,ET_i,ET_d));
check::globals(array(efoovar,foovar,cfoovar,a_sfoovar,a_foovar,a_efoovar));

$a = new A();
try {
    $a->foo();
} catch (Exception $e) {
    check::equal($e->getMessage(), 'C++ E1 exception thrown', '');
}

try {
    $a->bar();
} catch (Exception $e) {
    check::equal($e->getMessage(), 'C++ E2 exception thrown', '');
}

try {
    $a->foobar();
} catch (Exception $e) {
    check::equal($e->getMessage(), 'postcatch unknown', '');
}

try {
    $a->barfoo(1);
} catch (Exception $e) {
    check::equal($e->getMessage(), 'C++ E1 exception thrown', '');
}

try {
    $a->barfoo(2);
} catch (Exception $e) {
    check::equal($e->getMessage(), 'C++ E2 * exception thrown', '');
}
?>
