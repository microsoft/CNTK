<?php

	require "example.php";

	# First create some objects using the pointer library.

	print "Testing the pointer library\n";

	$a = 37.145;
	$b = 42.555;
	$c = "";  // $c must be defined and not null.

	print "	a = $a\n";
	print "	b = $b\n";
	print "	c = $c\n";

	# Call the add() function wuth some pointers
	add($a,$b,$c);

	print "	$a + $b = $c\n";

	# Now try the typemap library
	# This should be much easier. Now how it is no longer
	# necessary to manufacture pointers.

	print "Trying the typemap library\n";
	$r = sub(37,42);
	print "	37 - 42 = $r\n";

	# Now try the version with multiple return values
	# print "Testing multiple return values\n";
	# ($q,$r) = divide(42,37);
	# print "	42/37 = $q remainder $r\n";

?>
