<? 

require "tests.php";
require "virtual_vs_nonvirtual_base.php";

$fail = new SimpleClassFail();
$work = new SimpleClassWork();

check::equal($work->getInner()->get(), $fail->getInner()->get(), "should both be 10");

?>
