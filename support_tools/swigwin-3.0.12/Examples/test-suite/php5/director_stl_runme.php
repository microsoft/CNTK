<?php

require "tests.php";
require "director_stl.php";

// No new functions
check::functions(array(foo_bar,foo_ping,foo_pong,foo_tping,foo_tpong,foo_pident,foo_vident,foo_vsecond,foo_tpident,foo_tvident,foo_tvsecond,foo_vidents,foo_tvidents));
// No new classes
check::classes(array(Foo));
// now new vars
check::globals(array());

class MyFoo extends Foo {
  function ping($s) {
    return "MyFoo::ping():" . $s;
  }

  function pident($arg) {
    return $arg;
  }

  function vident($v) {
    return $v;
  }

  function vidents($v) {
    return $v;
  }

  function vsecond($v1, $v2) {
    return $v2;
  }
}

$a = new MyFoo();

$a->tping("hello");
$a->tpong("hello");

# TODO: automatic conversion between PHP arrays and std::pair or 
# std::vector is not yet implemented.
/*$p = array(1, 2);
$a->pident($p);
$v = array(3, 4);
$a->vident($v);

$a->tpident($p);
$a->tvident($v);

$v1 = array(3, 4);
$v2 = array(5, 6);

$a->tvsecond($v1, $v2);

$vs = array("hi", "hello");
$vs;
$a->tvidents($vs);*/

check::done();
?>
