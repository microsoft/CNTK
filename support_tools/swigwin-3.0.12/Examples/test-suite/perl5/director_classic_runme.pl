use strict;
use warnings;
use Test::More tests => 41;
BEGIN { use_ok('director_classic') }
require_ok('director_classic');

{
  package TargetLangPerson;
  use base 'director_classic::Person';
  sub id { return 'TargetLangPerson' }
}

{
  package TargetLangChild;
  use base 'director_classic::Child';
  sub id { return 'TargetLangChild' }
}

{
  package TargetLangGrandChild; 
  use base 'director_classic::GrandChild';
  sub id { return 'TargetLangGrandChild' }
}

# Semis - don't override id() in target language
{
  package TargetLangSemiPerson;
  use base 'director_classic::Person';
  # No id() override
}

{
  package TargetLangSemiChild;
  use base 'director_classic::Child';
  # No id() override
}

{
  package TargetLangSemiGrandChild;
  use base 'director_classic::GrandChild';
  # No id() override
}

# Orphans - don't override id() in C++
{
  package TargetLangOrphanPerson;
  use base 'director_classic::OrphanPerson';
  sub id { return "TargetLangOrphanPerson" }
}

{
  package TargetLangOrphanChild;
  use base 'director_classic::OrphanChild';
  sub id { return "TargetLangOrphanChild" }
}

sub check { my($person, $expected) = @_;
  # Normal target language polymorphic call
  is($person->id(), $expected, "$expected from Perl");

  # Polymorphic call from C++
  my $caller = director_classic::Caller->new();
  $caller->setCallback($person);
  is($caller->call(), $expected, "$expected from C++");

  # Polymorphic call of object created in target language and passed to C++ and back again
  my $baseclass = $caller->baseClass();
  is($baseclass->id(), $expected, "$expected after bounce");

  $caller->resetCallback();
}

my $person;

$person = director_classic::Person->new();
check($person, "Person");
undef $person;

$person = director_classic::Child->new();
check($person, "Child");
undef $person;

$person = director_classic::GrandChild->new();
check($person, "GrandChild"); 
undef $person;

$person = TargetLangPerson->new();
check($person, "TargetLangPerson"); 
undef $person;

$person = TargetLangChild->new();
check($person, "TargetLangChild"); 
undef $person;

$person = TargetLangGrandChild->new();
check($person, "TargetLangGrandChild"); 
undef $person;

# Semis - don't override id() in target language
$person = TargetLangSemiPerson->new();
check($person, "Person"); 
undef $person;

$person = TargetLangSemiChild->new();
check($person, "Child"); 
undef $person;

$person = TargetLangSemiGrandChild->new();
check($person, "GrandChild"); 
undef $person;

# Orphans - don't override id() in C++
$person = director_classic::OrphanPerson->new();
check($person, "Person");
undef $person;

$person = director_classic::OrphanChild->new();
check($person, "Child");
undef $person;

$person = TargetLangOrphanPerson->new();
check($person, "TargetLangOrphanPerson"); 
undef $person;

$person = TargetLangOrphanChild->new();
check($person, "TargetLangOrphanChild"); 
undef $person;

