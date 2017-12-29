# file: runme.tcl
# Test various properties of classes defined in separate modules

puts "Testing the %import directive"
catch { load ./base[info sharedlibextension] base}
catch { load ./foo[info sharedlibextension] foo}
catch { load ./bar[info sharedlibextension] bar}
catch { load ./spam[info sharedlibextension] spam}

# Create some objects

puts "Creating some objects"

set a [Base]
set b [Foo]
set c [Bar]
set d [Spam]

# Try calling some methods
puts "Testing some methods"
puts "Should see 'Base::A' ---> "
$a A
puts "Should see 'Base::B' ---> "
$a B

puts  "Should see 'Foo::A' ---> "
$b A
puts  "Should see 'Foo::B' ---> "
$b B

puts  "Should see 'Bar::A' ---> "
$c A
puts  "Should see 'Bar::B' ---> "
$c B

puts  "Should see 'Spam::A' ---> "
$d A
puts  "Should see 'Spam::B' ---> "
$d B

# Try some casts

puts "\nTesting some casts\n"

Base x -this [$a toBase]
puts "Should see 'Base::A' ---> "
x A
puts "Should see 'Base::B' ---> "
x B
rename x ""

Base x -this [$b toBase]
puts "Should see 'Foo::A' ---> "
x A
puts  "Should see 'Base::B' ---> "
x B
rename x ""

Base x -this [$c toBase]
puts  "Should see 'Bar::A' ---> "
x A
puts  "Should see 'Base::B' ---> "
x B
rename x ""

Base x -this [$d toBase]
puts  "Should see 'Spam::A' ---> "
x A
puts  "Should see 'Base::B' ---> "
x B
rename x ""

Bar x -this [$d toBar]
puts  "Should see 'Bar::B' ---> "
x B
rename x ""

puts "\nTesting some dynamic casts\n"
Base x -this [$d toBase]

puts  "Spam -> Base -> Foo : "
set y [Foo_fromBase [x cget -this]]
if {$y != "NULL"} {
      puts "bad swig"
} {
      puts "good swig"
}

puts  "Spam -> Base -> Bar : "
set y [Bar_fromBase [x cget -this]]
if {$y != "NULL"} {
      puts "good swig"
} {
      puts "bad swig"
}
      
puts  "Spam -> Base -> Spam : "
set y [Spam_fromBase [x cget -this]]
if {$y != "NULL"} {
      puts "good swig"
} {
      puts "bad swig"
}

puts  "Foo -> Spam : "
set y [Spam_fromBase $b]
if {$y != "NULL"} {
      puts "bad swig"
} {
      puts "good swig"
}



