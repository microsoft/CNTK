#!/usr/bin/perl

use strict;

my $modsize = 399; #adjust it so you can have a smaller or bigger hugemod 

my $runme = shift @ARGV;

open HEADER, ">hugemod.h" or die "error";
open TEST, ">$runme" or die "error";
open I1, ">hugemod_a.i" or die "error";
open I2, ">hugemod_b.i" or die "error";

print TEST "import hugemod_a\n";
print TEST "import hugemod_b\n";

print I1 "\%module hugemod_a;\n";
print I1 "\%include \"hugemod.h\";\n";
print I1 "\%{ #include \"hugemod.h\" \%}\n";

print I2 "\%module hugemod_b;\n";
print I2 "\%import \"hugemod.h\";\n";
print I2 "\%{ #include \"hugemod.h\" \%}\n";
print I2 "\%inline \%{\n";

my $i;

for ($i = 0; $i < $modsize; $i++) {
  my $t = $i * 4;
  print HEADER "class type$i { public: int a; };\n";
  print I2 "class dtype$i : public type$i { public: int b; };\n";
  
  print TEST "c = hugemod_a.type$i()\n";
  print TEST "c.a = $t\n";
  print TEST "if c.a != $t:\n";
  print TEST "    raise RuntimeError\n";

  print TEST "c = hugemod_b.dtype$i()\n";
  print TEST "c.a = $t\n";
  print TEST "if c.a != $t:\n";
  print TEST "    raise RuntimeError\n";
  
  $t = -$t;
  
  print TEST "c.b = $t\n";
  print TEST "if c.b != $t:\n";
  print TEST "    raise RuntimeError\n\n";
}

print I2 "\%}\n";

close HEADER;
close TEST;
close I1;
close I2;
