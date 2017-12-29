<?php

require "tests.php";
require "director_abstract.php";

// No new functions
check::functions(array(foo_ping,foo_pong,example0_getxsize,example0_color,example0_get_color,example1_getxsize,example1_color,example1_get_color,example2_getxsize,example2_color,example2_get_color,example4_getxsize,example4_color,example4_get_color,example3_i_color,example3_i_get_color,g,a_f));
// No new classes
check::classes(array(director_abstract,Foo,Example0,Example1,Example2,Example4,Example3_i,A));
// now new vars
check::globals(array());

class MyFoo extends Foo {
  function ping() {
    return "MyFoo::ping()";
  }
}

$a = new MyFoo();

check::equal($a->ping(), "MyFoo::ping()", "MyFoo::ping failed");

check::equal($a->pong(), "Foo::pong();MyFoo::ping()", "MyFoo::pong failed");

class MyExample1 extends Example1 {
  function Color($r, $g, $b) {
    return $r;
  }
}

class MyExample2 extends Example1 {
  function Color($r, $g, $b) {
    return $g;
  }
}

class MyExample3 extends Example1 {
  function Color($r, $g, $b) {
    return $b;
  }
}

$me1 = new MyExample1();
check::equal($me1->Color(1, 2, 3), 1, "Example1_get_color failed");

$me2 = new MyExample2(1, 2);
check::equal($me2->Color(1, 2, 3), 2, "Example2_get_color failed");

$me3 = new MyExample3();
check::equal($me3->Color(1, 2, 3), 3, "Example3_get_color failed");

$class = new ReflectionClass('Example1');
check::equal($class->isAbstract(), true, "Example1 abstractness failed");

$class = new ReflectionClass('Example2');
check::equal($class->isAbstract(), true, "Example2 abstractness failed");

$class = new ReflectionClass('Example3_i');
check::equal($class->isAbstract(), true, "Example3_i abstractness failed");

check::done();
?>
