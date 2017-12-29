<?php

require "tests.php";
require "director_nested.php";

// No new functions
check::functions(array(foo_int_advance,foo_int_do_advance,bar_step,bar_do_advance,bar_do_step,foobar_int_get_value,foobar_int_get_name,foobar_int_name,foobar_int_get_self,foobar_int_do_advance,foobar_int_do_step));
// No new classes
check::classes(array(Foo_int,Bar,FooBar_int));
// now new vars
check::globals(array());

class A extends FooBar_int {
  function do_step() {
    return "A::do_step;";
  }

  function get_value() {
    return "A::get_value";
  }
}

$a = new A();
check::equal($a->step(), "Bar::step;Foo::advance;Bar::do_advance;A::do_step;", "Bad A virtual resolution");

class B extends FooBar_int {
  function do_advance() {
    return "B::do_advance;" . $this->do_step();
  }

  function do_step() {
    return "B::do_step;";
  }

  function get_value() {
    return 1;
  }
}

$b = new B();

check::equal($b->step(), "Bar::step;Foo::advance;B::do_advance;B::do_step;", "Bad B virtual resolution");

class C extends FooBar_int {
  function do_advance() {
    return "C::do_advance;" . parent::do_advance();
  }

  function do_step() {
    return "C::do_step;";
  }

  function get_value() {
    return 2;
  }

  function get_name() {
    return parent::get_name() . " hello";
  }
}

$cc = new C();
# TODO: Currently we do not track the dynamic type of returned 
# objects, so we skip the get_self() call.
#$c = Foobar_int::get_self($cc);
$c = $cc;
$c->advance();

check::equal($c->get_name(), "FooBar::get_name hello", "get_name failed");

check::equal($c->name(), "FooBar::get_name hello", "name failed");

check::done();
?>
