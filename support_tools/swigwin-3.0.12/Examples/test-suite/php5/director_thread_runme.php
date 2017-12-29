<?php

require "tests.php";
require "director_thread.php";

# Fails in a ZTS-build of PHP - see: https://github.com/swig/swig/pull/155
exit(0);

// No new functions
check::functions(array(millisecondsleep,foo_stop,foo_run,foo_do_foo));
// No new classes
check::classes(array(director_thread,Foo));
// now new vars
check::globals(array(foo_val));

class Derived extends Foo {
  function do_foo() {
    $this->val = $this->val - 1;
  }
}

$d = new Derived();
$d->run();

if ($d->val >= 0) {
  check::fail($d->val);
}

$d->stop();

check::done();
?>
