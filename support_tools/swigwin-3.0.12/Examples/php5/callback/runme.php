<?php

# This file illustrates the cross language polymorphism using directors.

require("example.php");

# Class, which overwrites Callback::run().

class PhpCallback extends Callback {
  function run() {
    print "PhpCallback.run()\n";
  }
};

# Create an Caller instance

$caller = new Caller();

# Add a simple C++ callback (caller owns the callback, so
# we disown it first by clearing the .thisown flag).

print "Adding and calling a normal C++ callback\n";
print "----------------------------------------\n";

$callback = new Callback();
$callback->thisown = 0;
$caller->setCallback($callback);
$caller->call();
$caller->delCallback();

print "\n";
print "Adding and calling a PHP callback\n";
print "------------------------------------\n";

# Add a PHP callback.

$callback = new PhpCallback();
$callback->thisown = 0;
$caller->setCallback($callback);
$caller->call();
$caller->delCallback();

# All done.

print "php exit\n";

?>
