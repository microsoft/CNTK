<?php

require "tests.php";
require "director_detect.php";

// No new functions
check::functions(array(foo_cloner,foo_get_value,foo_get_class,foo_just_do_it,bar_baseclass,bar_cloner,bar_get_value,bar_get_class,bar_just_do_it));
// No new classes
check::classes(array(A,Foo,Bar));
// now new vars
check::globals(array());

class MyBar extends Bar {
  function __construct($val = 2) {
    parent::__construct();
    $this->val = $val;
  }

  function get_value() {
    $this->val = $this->val + 1;
    return $this->val;
  }

  function get_class() {
    $this->val = $this->val + 1;
    return new A();
  }

  function just_do_it() {
    $this->val = $this->val + 1;
  }

  /* clone is a reserved keyword */
  function clone_() {
    return new MyBar($this->val);
  }
}

$b = new MyBar();

$f = $b->baseclass();

$v = $f->get_value();
$a = $f->get_class();
$f->just_do_it();

$c = $b->clone_();
$vc = $c->get_value();

check::equal($v, 3, "f: Bad virtual detection");
check::equal($b->val, 5, "b: Bad virtual detection");
check::equal($vc, 6, "c: Bad virtual detection");

check::done();
?>
