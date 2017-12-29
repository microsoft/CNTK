<?php

require "tests.php";
require "director_extend.php";

// No new functions
check::functions(array(spobject_getfoobar,spobject_dummy,spobject_exceptionmethod));
// No new classes
check::classes(array(SpObject));
// now new vars
check::globals(array());

class MyObject extends SpObject{
  function getFoo() {
    return 123;
  }
}

$m = new MyObject();
check::equal($m->dummy(), 666, "1st call");
check::equal($m->dummy(), 666, "2st call"); // Locked system

check::done();
?>
