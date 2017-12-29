<?php

	require "example.php";
	echo "\nVariables (values printed from C)\n";

	print_vars();

	echo "Variables (values printed from PHP)\n";

	echo "ivar	= ".ivar_get()."\n";
	echo "svar	= ".svar_get()."\n";
	echo "lvar	= ".lvar_get()."\n";
	echo "uivar	= ".uivar_get()."\n";
	echo "usvar	= ".usvar_get()."\n";
	echo "ulvar	= ".ulvar_get()."\n";
	echo "scvar	= ".scvar_get()."\n";
	echo "ucvar	= ".ucvar_get()."\n";
	echo "cvar	= ".cvar_get()."\n";
	echo "fvar	= ".fvar_get()."\n";
	echo "dvar	= ".dvar_get()."\n";
	echo "strvar	= ".strvar_get()."\n";
	echo "cstrvar	= ".cstrvar_get()."\n";
	echo "iptrvar	= ".iptrvar_get()."\n";
	echo "name	= \"".name_get()."\"\n";
	echo "ptptr	= ".ptptr_get() , point_print(ptptr_get()) , "\n";
	echo "pt	= ".pt_get(), point_print(pt_get()) , "\n";

	/* Try to set the values of some global variables */
$a = "42.14";

	ivar_set($a);
echo "a = $a\n";
	svar_set(-31000);
	lvar_set(65537);
	uivar_set(123456);
	usvar_set(61000);
	ulvar_set(654321);
	scvar_set(-13);
	ucvar_set(251);
	cvar_set("S");
	fvar_set(3.14159);
	dvar_set(2.1828);
	strvar_set("Hello World");
	iptrvar_set(new_int(37));
	ptptr_set(new_point(37,42));
	name_set("B");

	echo "Variables (values printed from PHP)\n";

	echo "ivar	= ".ivar_get()."\n";
	echo "svar	= ".svar_get()."\n";
	echo "lvar	= ".lvar_get()."\n";
	echo "uivar	= ".uivar_get()."\n";
	echo "usvar	= ".usvar_get()."\n";
	echo "ulvar	= ".ulvar_get()."\n";
	echo "scvar	= ".scvar_get()."\n";
	echo "ucvar	= ".ucvar_get()."\n";
	echo "cvar	= ".cvar_get()."\n";
	echo "fvar	= ".fvar_get()."\n";
	echo "dvar	= ".dvar_get()."\n";
	echo "strvar	= ".strvar_get()."\n";
	echo "cstrvar	= ".cstrvar_get()."\n";
	echo "iptrvar	= ".iptrvar_get()."\n";
	echo "name	= ".name_get()."\n";
	echo "ptptr	= ".ptptr_get() , point_print(ptptr_get()) , "\n";
	echo "pt	= ".pt_get(), point_print(pt_get()) , "\n";

	echo "\nVariables (values printed from C)\n";

	print_vars();

	echo "\nI'm going to try and update a structure variable.\n";

	pt_set(ptptr_get());

	echo "The new value is \n";

	pt_print();

	echo "You should see the value", point_print(ptptr_get()), "\n";

	echo "\nNow I'm going to try and modify some read only variables\n";

	echo "Trying to set 'path'\n";

	//path_set("Whoa!");
	echo "Path = ".path_get()."\n";

	echo "Trying to set 'status'\n";

	/* And this */
	//status_set(0);
	echo "Status = ".status_get()."\n";

?>

