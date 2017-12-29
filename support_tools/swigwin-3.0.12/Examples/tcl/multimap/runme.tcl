# file: runme.tcl
# Try to load as a dynamic module.

catch { load ./example[info sharedlibextension] example}

# Call our gcd() function
set x 42
set y 105
set g [gcd $x $y]
puts "The gcd of $x and $y is $g"

# call the gcdmain
gcdmain "gcdmain 42 105"


# call count
set c [count "Hello World" l]
puts $c

# call capitalize

set c [capitalize "helloworld"]
puts $c

