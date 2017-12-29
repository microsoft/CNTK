<?

// Load module and PHP classes.
include("example.php");

echo "Got new object\n";
echo "Got string $s and value $x \n";

$s = new Sync();
echo "Got new object\n";

$s->printer();

?>

