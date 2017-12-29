<?php

require "tests.php";
require "director_string.php";

// No new functions
check::functions(array(a_get_first,a_call_get_first,a_string_length,a_process_text,a_call_process_func,stringvector_size,stringvector_is_empty,stringvector_clear,stringvector_push,stringvector_pop,stringvector_capacity,stringvector_reserve));
// No new classes
check::classes(array(A,StringVector));
// now new vars
check::globals(array(a,a_call,a_m_strings,stringvector));

class B extends A {
  function get_first() {
    return parent::get_first() . " world!";
  }

  function process_text($string) {
    parent::process_text($string);
    $this->smem = "hello";
  }
}

$b = new B("hello");

$b->get(0);
check::equal($b->get_first(),"hello world!", "get_first failed");

$b->call_process_func();

check::equal($b->smem, "hello", "smem failed");

check::done();
?>
