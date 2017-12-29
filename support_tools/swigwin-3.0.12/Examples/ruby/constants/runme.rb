# file: runme.rb

require 'example'

print "ICONST  = ", Example::ICONST,  " (should be 42)\n"
print "FCONST  = ", Example::FCONST,  " (should be 2.1828)\n"
print "CCONST  = ", Example::CCONST,  " (should be 'x')\n"
print "CCONST2 = ", Example::CCONST2, " (this should be on a new line)\n"
print "SCONST  = ", Example::SCONST,  " (should be 'Hello World')\n"
print "SCONST2 = ", Example::SCONST2, " (should be '\"Hello World\"')\n"
print "EXPR    = ", Example::EXPR,    " (should be 48.5484)\n"
print "iconst  = ", Example::Iconst,  " (should be 37)\n"
print "fconst  = ", Example::Fconst,  " (should be 3.14)\n"

begin
  print "EXTERN = ", Example::EXTERN, " (Arg! This shouldn't print anything)\n"
rescue NameError
  print "EXTERN isn't defined (good)\n"
end

begin
  print "FOO    = ", Example::FOO, " (Arg! This shouldn't print anything)\n"
rescue NameError
  print "FOO isn't defined (good)\n"
end
