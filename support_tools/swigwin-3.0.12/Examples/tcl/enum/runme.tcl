# file: runme.tcl

catch { load ./example[info sharedlibextension] example}

# ----- Object creation -----

# Print out the value of some enums
puts "*** color ***"
puts "    RED    = $RED"
puts "    BLUE   = $BLUE"
puts "    GREEN  = $GREEN"

puts "\n*** Foo::speed ***"
puts "    Foo_IMPULSE   = $Foo_IMPULSE"
puts "    Foo_WARP      = $Foo_WARP"
puts "    Foo_LUDICROUS = $Foo_LUDICROUS"


puts "\nTesting use of enums with functions\n"

enum_test $RED   $Foo_IMPULSE
enum_test $BLUE  $Foo_WARP
enum_test $GREEN $Foo_LUDICROUS
enum_test 1234   5678

puts "\nTesting use of enum with class method"
Foo f

f enum_test $Foo_IMPULSE
f enum_test $Foo_WARP
f enum_test $Foo_LUDICROUS

