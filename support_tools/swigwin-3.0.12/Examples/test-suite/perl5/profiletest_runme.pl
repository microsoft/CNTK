use profiletest;
$a = profiletestc::new_A();
$b = profiletestc::new_B();

for ($i = 0; $i < 100000; $i++) {
    $a = profiletestc::B_fn($b, $a);
}

