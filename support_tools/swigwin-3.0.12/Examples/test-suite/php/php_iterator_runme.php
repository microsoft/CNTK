<?php

require "tests.php";
require "php_iterator.php";

check::functions(array(myiterator_rewind,myiterator_key,myiterator_current,myiterator_next,myiterator_valid));
check::classes(array(MyIterator));
// No new global variables.
check::globals(array());

$s = '';
foreach (new MyIterator(1, 6) as $i) {
  $s .= $i;
}
check::equal($s, '12345', 'Simple iteration failed');

$s = '';
foreach (new MyIterator(2, 5) as $k => $v) {
  $s .= "($k=>$v)";
}
check::equal($s, '(0=>2)(1=>3)(2=>4)', 'Simple iteration failed');

check::done();
?>
