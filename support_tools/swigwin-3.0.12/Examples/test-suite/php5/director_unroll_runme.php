<?php

require "tests.php";
require "director_unroll.php";

// No new functions
check::functions(array(foo_ping,foo_pong));
// No new classes
check::classes(array(Foo,Bar));
// now new vars
check::globals(array(bar));

class MyFoo extends Foo {
  function ping() {
    return "MyFoo::ping()";
  }
}

$a = new MyFoo();

$b = new Bar();

$b->set($a);
$c = $b->get();

check::equal($a->this, $c->this, "this failed");

check::done();
?>
