# Operator overloading example

catch { load ./example[info sharedlibextension] example}

set a [Complex -args 2 3]
set b [Complex -args -5 10]

puts "a   = $a [$a str]"
puts "b   = $b [$b str]"

set c [$a + $b]
puts "c   = $c [$c str]"

set d [$a * $b]
puts "a*b  = [$d str]"

# Alternative calling convention
set e [Complex_- $a $c]
puts "a-c  = [Complex_str $e]"

set f [new_ComplexCopy $e]
puts "f    = [$f str]"

# Call assignment operator
$c = $f
puts "c    = [$c str]"

