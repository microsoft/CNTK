<?php

	require "example.php";


	$v = new_vector();
        vector_x_set($v,1.0);
        vector_y_set($v,2.0);
        vector_z_set($v,3.0);

	$w = new_vector();
        vector_x_set($w,10.0);
        vector_y_set($w,11.0);
        vector_z_set($w,12.0);

	echo "I just created the following vector\n";
	vector_print($v);
	vector_print($w);

	echo "\nNow I'm going to compute the dot product\n";

	$d = dot_product($v, $w);

	echo "dot product = $d (should be 68)\n";

	echo "\nNow I'm going to add the vectors together\n";

        $r = new_vector();
	vector_add($v, $w, $r);

	vector_print($r);

	echo "The value should be (11,13,15)\n";

	echo "\nNow I'm going to clean up the return result\n";

#	free($r);

	echo "Good\n";

?>


