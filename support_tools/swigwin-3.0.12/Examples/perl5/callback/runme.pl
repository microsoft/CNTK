# file: runme.pl

# This file illustrates the cross language polymorphism using directors.

use example;


{
  package PlCallback;
  use base 'example::Callback';
  sub run {
    print "PlCallback->run()\n";
  }
}

# Create an Caller instance

$caller = example::Caller->new();

# Add a simple C++ callback (caller owns the callback, so
# we disown it first by clearing the .thisown flag).

print "Adding and calling a normal C++ callback\n";
print "----------------------------------------\n";

$callback = example::Callback->new();
$callback->DISOWN();
$caller->setCallback($callback);
$caller->call();
$caller->delCallback();

print "\n";
print "Adding and calling a Perl callback\n";
print "----------------------------------\n";

# Add a Perl callback (caller owns the callback, so we
# disown it first by calling DISOWN).

$callback = PlCallback->new();
$callback->DISOWN();
$caller->setCallback($callback);
$caller->call();
$caller->delCallback();

# All done.

print "\n";
print "perl exit\n";
