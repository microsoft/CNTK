# file: runme.rb

require 'example'

# ----- Object creation -----

# Print out the value of some enums
print "*** color ***\n"
print "    RED    = #{Example::RED}\n"
print "    BLUE   = #{Example::BLUE}\n"
print "    GREEN  = #{Example::GREEN}\n"

print "\n*** Foo::speed ***\n"
print "    Foo::IMPULSE   = #{Example::Foo::IMPULSE}\n"
print "    Foo::WARP      = #{Example::Foo::WARP}\n"
print "    Foo::LUDICROUS = #{Example::Foo::LUDICROUS}\n"

print "\nTesting use of enums with functions\n\n"

Example::enum_test(Example::RED, Example::Foo::IMPULSE)
Example::enum_test(Example::BLUE,  Example::Foo::WARP)
Example::enum_test(Example::GREEN, Example::Foo::LUDICROUS)
Example::enum_test(1234, 5678)

print "\nTesting use of enum with class method\n"
f = Example::Foo.new()

f.enum_test(Example::Foo::IMPULSE)
f.enum_test(Example::Foo::WARP)
f.enum_test(Example::Foo::LUDICROUS)
