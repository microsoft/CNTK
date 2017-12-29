# file: runme.tcl

catch { load ./example[info sharedlibextension] example}

puts "ICONST  = $ICONST (should be 42)"
puts "FCONST  = $FCONST (should be 2.1828)"
puts "CCONS T = $CCONST (should be 'x')"
puts "CCONST2 = $CCONST2 (this should be on a separate line)"
puts "SCONST  = $SCONST (should be 'Hello World')"
puts "SCONST2 = $SCONST2 (should be '\"Hello World\"')" 
puts "EXPR    = $EXPR (should be 48.5484)"
puts "iconst  = $iconst (should be 37)"
puts "fconst  = $fconst (should be 3.14)"

if { [catch {
    puts "EXTERN = $EXTERN (Arg! This shouldn't print anything)"
}]} {
    puts "EXTERN isn't defined (good)"
}

if { [catch {
    puts "FOO    = $FOO (Arg! This shouldn't print anything)"
}]} {
    puts "FOO isn't defined (good)"
}

