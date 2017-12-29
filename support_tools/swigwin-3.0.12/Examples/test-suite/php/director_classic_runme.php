<?php

require "tests.php";
require "director_classic.php";

// No new functions
check::functions(array(being_id,person_id,child_id,grandchild_id,caller_delcallback,caller_setcallback,caller_resetcallback,caller_call,caller_baseclass));
// No new classes
check::classes(array(Being,Person,Child,GrandChild,OrphanPerson,OrphanChild,Caller));
// now new vars
check::globals(array());

class TargetLangPerson extends Person {
  function id() {
    $identifier = "TargetLangPerson";
    return $identifier;
  }
}

class TargetLangChild extends Child {
  function id() {
    $identifier = "TargetLangChild";
    return $identifier;
  }
}

class TargetLangGrandChild extends GrandChild {
  function id() {
    $identifier = "TargetLangGrandChild";
    return $identifier;
  }
}

# Semis - don't override id() in target language
class TargetLangSemiPerson extends Person {
  # No id() override
}

class TargetLangSemiChild extends Child {
  # No id() override
}

class TargetLangSemiGrandChild extends GrandChild {
  # No id() override
}

# Orphans - don't override id() in C++
class TargetLangOrphanPerson extends OrphanPerson {
  function id() {
    $identifier = "TargetLangOrphanPerson";
    return $identifier;
  }
}

class TargetLangOrphanChild extends OrphanChild {
  function id() {
    $identifier = "TargetLangOrphanChild";
    return $identifier;
  }
}

function mycheck($person, $expected) {
  $debug = 0;
  # Normal target language polymorphic call
  $ret = $person->id();
  if ($debug)
    print $ret . "\n";
  check::equal($ret, $expected, "#1 failed");

  # Polymorphic call from C++
  $caller = new Caller();
  $caller->setCallback($person);
  $ret = $caller->call();
  if ($debug)
    print $ret . "\n";
  check::equal($ret, $expected, "#2 failed");

  # Polymorphic call of object created in target language and passed to 
  # C++ and back again
  $baseclass = $caller->baseClass();
  $ret = $baseclass->id();
  if ($debug)
    print $ret . "\n";
  # TODO: Currently we do not track the dynamic type of returned 
  # objects, so in case it's possible that the dynamic type is not equal 
  # to the static type, we skip this check.
  if (get_parent_class($person) === false)
    check::equal($ret, $expected, "#3 failed");

  $caller->resetCallback();
  if ($debug)
    print "----------------------------------------\n";
}

$person = new Person();
mycheck($person, "Person");
unset($person);

$person = new Child();
mycheck($person, "Child");
unset($person);

$person = new GrandChild();
mycheck($person, "GrandChild");
unset($person);

$person = new TargetLangPerson();
mycheck($person, "TargetLangPerson");
unset($person);

$person = new TargetLangChild();
mycheck($person, "TargetLangChild");
unset($person);

$person = new TargetLangGrandChild();
mycheck($person, "TargetLangGrandChild");
unset($person);

# Semis - don't override id() in target language
$person = new TargetLangSemiPerson();
mycheck($person, "Person");
unset($person);

$person = new TargetLangSemiChild();
mycheck($person, "Child");
unset($person);

$person = new TargetLangSemiGrandChild();
mycheck($person, "GrandChild");
unset($person);

# Orphans - don't override id() in C++
$person = new OrphanPerson();
mycheck($person, "Person");
unset($person);

$person = new OrphanChild();
mycheck($person, "Child");
unset($person);

$person = new TargetLangOrphanPerson();
mycheck($person, "TargetLangOrphanPerson");
unset($person);

$person = new TargetLangOrphanChild();
mycheck($person, "TargetLangOrphanChild");
unset($person);

check::done();
?>
