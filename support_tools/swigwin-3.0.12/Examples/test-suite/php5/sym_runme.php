<?php

require "tests.php";
require "sym.php";

// No new functions
check::functions(array());
// No new classes
check::classes(array(flim,flam));
// now new vars
check::globals(array());

$flim=new flim();
$flam=new flam();

check::equal($flim->hulahoops(),"flim-jam","flim()->hulahoops==flim-jam");
check::equal($flim->jar(),"flim-jar","flim()->jar==flim-jar");
check::equal($flam->jam(),"flam-jam","flam()->jam==flam-jam");
check::equal($flam->jar(),"flam-jar","flam()->jar==flam-jar");

check::done();
?>
