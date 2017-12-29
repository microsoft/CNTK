# file: runme.tcl

# This file illustrates the high level C++ interface.
# In this case C++ classes work kind of like Tk widgets

catch { load ./example[info sharedlibextension] example}

# ----- Object creation -----

puts "Creating some objects:"
Circle c 10
puts "    Created circle [c cget -this]"
Square s 10
puts "    Created square [s cget -this]"

# ----- Access a static member -----

puts "\nA total of $Shape_nshapes shapes were created"

# ----- Member data access -----

# Set the location of the object

c configure -x 20 -y 30
s configure -x -10 -y 5

puts "\nHere is their current position:"
puts "    Circle = ([c cget -x], [c cget -y])"
puts "    Square = ([s cget -x], [s cget -y])"

# ----- Call some methods -----

puts "\nHere are some properties of the shapes:"
foreach o "c s" {
      puts "    [$o cget -this]"
      puts "        area      = [$o area]"
      puts "        perimeter = [$o perimeter]"
}

# ----- Delete everything -----

puts "\nGuess I'll clean up now"

# Note: this invokes the virtual destructor
rename c ""
rename s ""

puts "$Shape_nshapes shapes remain"
puts "Goodbye"

