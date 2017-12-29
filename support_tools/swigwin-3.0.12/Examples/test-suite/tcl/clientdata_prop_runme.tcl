
if [ catch { load ./clientdata_prop_b[info sharedlibextension] clientdata_prop_b} err_msg ] {
  puts stderr "Could not load shared object:\n$err_msg"
  exit 1
}
if [ catch { load ./clientdata_prop_a[info sharedlibextension] clientdata_prop_a} err_msg ] {
  puts stderr "Could not load shared object:\n$err_msg"
  exit 1
}

A a
test_A a
test_tA a
test_t2A a
test_t3A a
a fA

B b
test_A b
test_tA b
test_t2A b
test_t3A b
test_B b
b fA
b fB

C c
test_A c
test_tA c
test_t2A c
test_t3A c
test_C c
c fA
c fC

D d
test_A d
test_tA d
test_t2A d
test_t3A d
test_D d
test_tD d
test_t2D d
d fA
d fD

set a2 [new_tA]
test_A $a2
test_tA $a2
test_t2A $a2
test_t3A $a2
$a2 fA

set a3 [new_t2A]
test_A $a3
test_tA $a3
test_t2A $a3
test_t3A $a3
$a3 fA

set a4 [new_t3A]
test_A $a4
test_tA $a4
test_t2A $a4
test_t3A $a4
$a4 fA

set d2 [new_tD]
test_A $d2
test_tA $d2
test_t2A $d2
test_t3A $d2
test_D $d2
test_tD $d2
test_t2D $d2
$d2 fA
$d2 fD

set d3 [new_t2D]
test_A $d3
test_tA $d3
test_t2A $d3
test_t3A $d3
test_D $d3
test_tD $d3
test_t2D $d3
$d3 fA
$d3 fD
