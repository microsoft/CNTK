<?php

require "tests.php";
require "abstract_inherit.php";

check::classes(array(Foo,Bar,Spam,NRFilter_i,NRRCFilter_i,NRRCFilterpro_i,NRRCFilterpri_i));
// This constructor attempt should fail as there isn't one
//$spam=new Spam();

//check::equal(0,$spam->blah(),"spam object method");
//check::equal(0,Spam::blah($spam),"spam class method");

check::done();
?>
