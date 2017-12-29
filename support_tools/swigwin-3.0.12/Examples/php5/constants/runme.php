<?php

require "example.php";

print "ICONST  = " . ICONST . " (should be 42)\n";
print "FCONST  = " . FCONST . " (should be 2.1828)\n";
print "CCONST  = " . CCONST . " (should be 'x')\n";
print "CCONST2 = " . CCONST2 . " (this should be on a new line)\n";
print "SCONST  = " . SCONST . " (should be 'Hello World')\n";
print "SCONST2 = " . SCONST2 . " (should be '\"Hello World\"')\n";
print "EXPR    = " . EXPR  . " (should be 48.5484)\n";
print "iconst  = " . iconst . " (should be 37)\n";
print "fconst  = " . fconst . " (should be 3.14)\n";

if (EXTERN!="EXTERN") {
    print "EXTERN = " . EXTERN . " (Arg! This shouldn't print anything)\n";
} else {
    print "EXTERN defaults to 'EXTERN', it probably isn't defined (good)\n";
}

if (FOO!="FOO") {
    print "FOO    = " . FOO . "(Arg! This shouldn't print anything)\n";
} else {
    print "FOO defaults to 'FOO', it probably isn't defined (good)\n";
}


?>
