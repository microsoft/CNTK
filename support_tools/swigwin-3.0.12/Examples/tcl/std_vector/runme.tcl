# file: runme.tcl

catch { load ./example[info sharedlibextension] example}

# Exercise IntVector

set iv [IntVector]
$iv push 1
$iv push 3
$iv push 5

puts "IntVector size:      [$iv size]   (should be 3)"
puts "IntVector average:   [average $iv] (should be 3.0)"
puts "IntVector pop:       [$iv pop]   (should be 5)"
puts "IntVector pop:       [$iv pop]   (should be 3)"
puts "IntVector get 0:     [$iv get 0]   (should be 1)"
puts ""

# Exercise DoubleVector

set dv [DoubleVector]
$dv push 2 
$dv push 4
$dv push 6
 
puts "DoubleVector size:   [$dv size]           (should be 3)"
puts "DoubleVector data:   [$dv get 0] [$dv get 1] [$dv get 2] (should be 2.0 4.0 6.0)"
halve_in_place $dv
puts "DoubleVector halved: [$dv get 0] [$dv get 1] [$dv get 2] (should be 1.0 2.0 3.0)"
puts ""

# Complain if unknown is called
rename unknown unknown_orig
proc unknown {args} {
  puts "ERROR: unknown called with: $args"
  uplevel 1 unknown_orig $args
}

puts "average \"1 2 3\": [average [list 1 2 3]]"

