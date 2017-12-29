# file: runme.pl

use example;

print "ICONST  = ", example::ICONST, " (should be 42)\n";
print "FCONST  = ", example::FCONST, " (should be 2.1828)\n";
print "CCONST  = ", example::CCONST, " (should be 'x')\n";
print "CCONST2 = ", example::CCONST2," (this should be on a new line)\n";
print "SCONST  = ", example::SCONST, " (should be 'Hello World')\n";
print "SCONST2 = ", example::SCONST2, " (should be '\"Hello World\"')\n";
print "EXPR    = ", example::EXPR,   " (should be 48.5484)\n";
print "iconst  = ", example::iconst, " (should be 37)\n";
print "fconst  = ", example::fconst, " (should be 3.14)\n";



