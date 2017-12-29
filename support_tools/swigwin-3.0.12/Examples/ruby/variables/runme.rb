# file: runme.rb

require 'example'

# Try to set the values of some global variables

Example.ivar   =  42
Example.svar   = -31000
Example.lvar   =  65537
Example.uivar  =  123456
Example.usvar  =  61000
Example.ulvar  =  654321
Example.scvar  =  -13
Example.ucvar  =  251
Example.cvar   =  "S"
Example.fvar   =  3.14159
Example.dvar   =  2.1828
Example.strvar =  "Hello World"
Example.iptrvar= Example.new_int(37)
Example.ptptr  = Example.new_Point(37,42)
Example.name   = "Bill"

# Now print out the values of the variables

puts "Variables (values printed from Ruby)"

puts "ivar      = #{Example.ivar}"
puts "svar      = #{Example.svar}"
puts "lvar      = #{Example.lvar}"
puts "uivar     = #{Example.uivar}"
puts "usvar     = #{Example.usvar}"
puts "ulvar     = #{Example.ulvar}"
puts "scvar     = #{Example.scvar}"
puts "ucvar     = #{Example.ucvar}"
puts "fvar      = #{Example.fvar}"
puts "dvar      = #{Example.dvar}"
puts "cvar      = #{Example.cvar}"
puts "strvar    = #{Example.strvar}"
puts "cstrvar   = #{Example.cstrvar}"
puts "iptrvar   = #{Example.iptrvar}"
puts "name      = #{Example.name}"
puts "ptptr     = #{Example.ptptr} (#{Example.Point_print(Example.ptptr)})"
puts "pt        = #{Example.pt} (#{Example.Point_print(Example.pt)})"

puts "\nVariables (values printed from C)"

Example.print_vars()

puts "\nNow I'm going to try and modify some read only variables";

puts "     Tring to set 'path'";
begin
  Example.path = "Whoa!"
  puts "Hey, what's going on?!?! This shouldn't work"
rescue NameError
  puts "Good."
end

puts "     Trying to set 'status'";
begin
  Example.status = 0
  puts "Hey, what's going on?!?! This shouldn't work"
rescue NameError
  puts "Good."
end


puts "\nI'm going to try and update a structure variable.\n"

Example.pt = Example.ptptr

puts "The new value is"
Example.pt_print()
puts "You should see the value #{Example.Point_print(Example.ptptr)}"



