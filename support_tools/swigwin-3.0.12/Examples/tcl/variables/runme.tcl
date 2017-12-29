# file: runme.tcl

catch { load ./example[info sharedlibextension] example}

# Try to set the values of some global variables

set ivar    42
set svar   -31000
set lvar    65537
set uivar   123456
set usvar   61000
set ulvar   654321
set scvar   -13
set ucvar   251
set cvar    "S"
set fvar    3.14159
set dvar    2.1828
set strvar  "Hello World"
set iptrvar [new_int 37]
set ptptr [new_Point 37 42]
set name    "Bill"

# Now print out the values of the variables

puts "Variables (values printed from Tcl)"

puts "ivar      = $ivar"
puts "svar      = $svar"
puts "lvar      = $lvar"
puts "uivar     = $uivar"
puts "usvar     = $usvar"
puts "ulvar     = $ulvar"
puts "scvar     = $scvar"
puts "ucvar     = $ucvar"
puts "fvar      = $fvar"
puts "dvar      = $dvar"
puts "cvar      = $cvar"
puts "strvar    = $strvar"
puts "cstrvar   = $cstrvar"
puts "iptrvar   = $iptrvar"
puts "name      = $name"
puts "ptptr     = $ptptr [Point_print $ptptr]"
puts "pt        = $pt [Point_print $pt]"

puts "\nVariables (values printed from C)"

print_vars

puts "\nNow I'm going to try and modify some read only variables";

puts "     Tring to set 'path'";
if { [catch {
    set path "Whoa!"
    puts "Hey, what's going on?!?! This shouldn't work"
}]} {
    puts "Good."
}

puts "     Trying to set 'status'";
if { [catch {
    set status 0
    puts "Hey, what's going on?!?! This shouldn't work"
}]} {
    puts "Good."
}

puts "\nI'm going to try and update a structure variable.\n"

set pt $ptptr

puts "The new value is"
pt_print
puts "You should see the value [Point_print $ptptr]"



