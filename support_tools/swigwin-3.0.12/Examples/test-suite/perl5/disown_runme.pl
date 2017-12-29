use disown;

if (1) {
    $a = new disown::A();
    $b = new disown::B();    
    $c = $b->acquire($a);   
}



