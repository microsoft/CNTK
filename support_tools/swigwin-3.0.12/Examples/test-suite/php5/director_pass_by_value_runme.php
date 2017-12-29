<?php

require "tests.php";
require "director_pass_by_value.php";

$passByVal = null;

class director_pass_by_value_Derived extends DirectorPassByValueAbstractBase {
  function virtualMethod($b) {
    global $passByVal;
    $passByVal = $b;
  }
}

# bug was the passByVal global object was destroyed after the call to virtualMethod had finished.
$caller = new Caller();
$caller->call_virtualMethod(new director_pass_by_value_Derived());
$ret = $passByVal->getVal();
if ($ret != 0x12345678) {
  check::fail("Bad return value, got " . dechex($ret));
}

check::done();
?>
