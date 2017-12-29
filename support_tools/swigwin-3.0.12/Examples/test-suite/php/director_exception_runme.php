<?php

require "tests.php";
require "director_exception.php";

// No new functions
check::functions(array(foo_ping,foo_pong,launder,bar_ping,bar_pong,bar_pang,returnalltypes_return_int,returnalltypes_return_double,returnalltypes_return_const_char_star,returnalltypes_return_std_string,returnalltypes_return_bar,returnalltypes_call_int,returnalltypes_call_double,returnalltypes_call_const_char_star,returnalltypes_call_std_string,returnalltypes_call_bar,is_python_builtin));
// No new classes
check::classes(array(director_exception,Foo,Exception1,Exception2,Base,Bar,ReturnAllTypes));
// now new vars
check::globals(array());

class MyException extends Exception {
  function __construct($a, $b) {
    $this->msg = $a . $b;
  }
}

class MyFoo extends Foo {
  function ping() {
    throw new Exception("MyFoo::ping() EXCEPTION");
  }
}

class MyFoo2 extends Foo {
  function ping() {
    return true;
  }
}

class MyFoo3 extends Foo {
  function ping() {
    throw new MyException("foo", "bar");
  }
}

# Check that the Exception raised by MyFoo.ping() is returned by 
# MyFoo.pong().
$ok = 0;
$a = new MyFoo();
# TODO: Currently we do not track the dynamic type of returned 
# objects, so we skip the launder() call.
#$b = director_exception::launder($a);
$b = $a;
try {
  $b->pong();
} catch (Exception $e) {
  $ok = 1;
  check::equal($e->getMessage(), "MyFoo::ping() EXCEPTION", "Unexpected error message #1");
}
check::equal($ok, 1, "Got no exception while expected one #1");

# Check that the director can return an exception which requires two 
# arguments to the constructor, without mangling it.
$ok = 0;
$a = new MyFoo3();
#$b = director_exception::launder($a);
$b = $a;
try {
  $b->pong();
} catch (Exception $e) {
  $ok = 1;
  check::equal($e->msg, "foobar", "Unexpected error message #2");
}
check::equal($ok, 1, "Got no exception while expected one #2");

try {
  throw new Exception2();
} catch (Exception2 $e2) {
}

try {
  throw new Exception1();
} catch (Exception1 $e1) {
}

// Check that we can throw exceptions from director methods (this didn't used
// to work in all cases, as the exception gets "set" in PHP and the method
// then returns PHP NULL, which the directorout template may fail to convert.

class Bad extends ReturnAllTypes {
  function return_int() { throw new Exception("bad int"); }
  function return_double() { throw new Exception("bad double"); }
  function return_const_char_star() { throw new Exception("bad const_char_star"); }
  function return_std_string() { throw new Exception("bad std_string"); }
  function return_Bar() { throw new Exception("bad Bar"); }
}

$bad = new Bad();

try {
    $bad->call_int();
    check::fail("Exception wasn't propagated from Bad::return_int()");
} catch (Exception $e) {
    check::equal($e->getMessage(), "bad int", "propagated exception incorrect");
}

try {
    $bad->call_double();
    check::fail("Exception wasn't propagated from Bad::return_double()");
} catch (Exception $e) {
    check::equal($e->getMessage(), "bad double", "propagated exception incorrect");
}

try {
    $bad->call_const_char_star();
    check::fail("Exception wasn't propagated from Bad::return_const_char_star()");
} catch (Exception $e) {
    check::equal($e->getMessage(), "bad const_char_star", "propagated exception incorrect");
}

try {
    $bad->call_std_string();
    check::fail("Exception wasn't propagated from Bad::return_std_string()");
} catch (Exception $e) {
    check::equal($e->getMessage(), "bad std_string", "propagated exception incorrect");
}

try {
    $bad->call_Bar();
    check::fail("Exception wasn't propagated from Bad::return_Bar()");
} catch (Exception $e) {
    check::equal($e->getMessage(), "bad Bar", "propagated exception incorrect");
}

check::done();
?>
