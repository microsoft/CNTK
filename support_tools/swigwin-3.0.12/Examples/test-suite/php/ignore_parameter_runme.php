<?php

require "tests.php";
require "ignore_parameter.php";

// New functions
check::functions(array(jaguar,lotus,tvr,ferrari,sportscars_daimler,sportscars_astonmartin,sportscars_bugatti,sportscars_lamborghini));
// New classes
check::classes(array(ignore_parameter,SportsCars,MiniCooper,MorrisMinor,FordAnglia,AustinAllegro));
// No new vars
check::globals(array());

check::equal(jaguar(2,3.4),"hello",'jaguar(2,3.4)=="hello"');
check::equal(lotus("eek",3.4),101,'lotus("eek",3.4)==101');
check::equal(tvr("eek",2),8.8,'tvr("eek",2)==8.8');
check::equal(ferrari(),101,'ferrari(2)==101');

$sc=new sportscars();
check::classname("sportscars",$sc);
check::equal($sc->daimler(2,3.4),"hello",'$sc->daimler(2,3.4)=="hello"');
check::equal($sc->astonmartin("eek",3.4),101,'$sc->mastonmartin("eek",3.4)==101');
check::equal($sc->bugatti("eek",2),8.8,'$sc->bugatti("eek",2)==8.8');
check::equal($sc->lamborghini(),101,'$sc->lamborghini(2)==101');

$mc=new minicooper(2,3.4);
check::classname("minicooper",$mc);

$mm=new morrisminor("eek",3.4);
check::classname("morrisminor",$mm);

$fa=new fordanglia("eek",2);
check::classname("fordanglia",$fa);

$aa=new austinallegro();
check::classname("austinallegro",$aa);

check::done();
?>
