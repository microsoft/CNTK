<?php

require "tests.php";
require "wrapmacro.php";

check::functions(array('guint16_swap_le_be_constant', 'maximum'));

check::equal(maximum(2.3, 2.4), 2.4, "maximum() doesn't work");
check::equal(guint16_swap_le_be_constant(0x1234), 0x3412, "GUINT16_SWAP_LE_BE_CONSTANT() doesn't work");

check::done();
?>
