<?php

require "tests.php";
require "director_frob.php";

// No new functions
check::functions(array(alpha_abs_method,bravo_abs_method,charlie_abs_method,ops_opint,ops_opintstarstarconst,ops_opintamp,ops_opintstar,ops_opconstintintstar,prims_ull,prims_callull,corecallbacks_on3dengineredrawn,corecallbacks_on3dengineredrawn2));
// No new classes
check::classes(array(Alpha,Bravo,Charlie,Delta,Ops,Prims,corePoint3d,coreCallbacks_On3dEngineRedrawnData,coreCallbacksOn3dEngineRedrawnData,coreCallbacks));
// now new vars
check::globals(array(corecallbacks_on3dengineredrawndata__eye,corecallbacks_on3dengineredrawndata__at,corecallbackson3dengineredrawndata__eye,corecallbackson3dengineredrawndata__at));

$foo = new Bravo();
$s = $foo->abs_method();

check::equal($s, "Bravo::abs_method()", "s failed");

check::done();
?>
