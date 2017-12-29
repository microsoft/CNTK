<?php

require "tests.php";
require "director_profile.php";

// No new functions
check::functions(array(b_fn,b_vfi,b_fi,b_fj,b_fk,b_fl,b_get_self,b_vfs,b_fs));
// No new classes
check::classes(array(A,B));
// now new vars
check::globals(array());

class MyB extends B {
  function vfi($a) {
    return $a+3;
  }
}

$a = new A();
$myb = new MyB();
$b = B::get_self($myb);

$i = 50000;
$a = 1;

while ($i) {
  $a = $b->fi($a); #1
  $a = $b->fi($a); #2
  $a = $b->fi($a); #3
  $a = $b->fi($a); #4
  $a = $b->fi($a); #5
  $a = $b->fi($a); #6
  $a = $b->fi($a); #7
  $a = $b->fi($a); #8
  $a = $b->fi($a); #9
  $a = $b->fi($a); #10
  $a = $b->fi($a); #1
  $a = $b->fi($a); #2
  $a = $b->fi($a); #3
  $a = $b->fi($a); #4
  $a = $b->fi($a); #5
  $a = $b->fi($a); #6
  $a = $b->fi($a); #7
  $a = $b->fi($a); #8
  $a = $b->fi($a); #9
  $a = $b->fi($a); #20
  $i -= 1;
}

print $a . "\n";

check::done();
?>
