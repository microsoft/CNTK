# file: runme.pl
# Test various properties of classes defined in separate modules

print "Testing the %import directive\n";
use baseclass;
use foo;
use bar;
use spam;

# Create some objects

print "Creating some objects\n";

$a = new baseclass::Base();
$b = new foo::Foo();
$c = new bar::Bar();
$d = new spam::Spam();

# Try calling some methods
print "Testing some methods\n";
print "Should see 'Base::A' ---> ";
$a->A();
print "Should see 'Base::B' ---> ";
$a->B();

print "Should see 'Foo::A' ---> ";
$b->A();
print "Should see 'Foo::B' ---> ";
$b->B();

print "Should see 'Bar::A' ---> ";
$c->A();
print "Should see 'Bar::B' ---> ";
$c->B();

print "Should see 'Spam::A' ---> ";
$d->A();
print "Should see 'Spam::B' ---> ";
$d->B();

# Try some casts

print "\nTesting some casts\n";

$x = $a->toBase();
print "Should see 'Base::A' ---> ";
$x->A();
print "Should see 'Base::B' ---> ";
$x->B();

$x = $b->toBase();
print "Should see 'Foo::A' ---> ";
$x->A();

print "Should see 'Base::B' ---> ";
$x->B();

$x = $c->toBase();
print "Should see 'Bar::A' ---> ";
$x->A();

print "Should see 'Base::B' ---> ";
$x->B();

$x = $d->toBase();
print "Should see 'Spam::A' ---> ";
$x->A();

print "Should see 'Base::B' ---> ";
$x->B();

$x = $d->toBar();
print "Should see 'Bar::B' ---> ";
$x->B();

print "\nTesting some dynamic casts\n";
$x = $d->toBase();

print " Spam -> Base -> Foo : ";
$y = foo::Foo::fromBase($x);
if ($y) {
    print "bad swig\n";
} else {
    print "good swig\n";
}

print " Spam -> Base -> Bar : ";
$y = bar::Bar::fromBase($x);
if ($y) {
    print "good swig\n";
} else {
    print "bad swig\n";
}
      
print " Spam -> Base -> Spam : ";
$y = spam::Spam::fromBase($x);
if ($y) {
    print "good swig\n";
} else {
    print "bad swig\n";
}

print " Foo -> Spam : ";
#print $b;
$y = spam::Spam::fromBase($b);
print $y;
if ($y) {
    print "bad swig\n";
} else {
    print "good swig\n";
}





