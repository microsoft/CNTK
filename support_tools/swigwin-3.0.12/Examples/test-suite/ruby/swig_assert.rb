#!/usr/bin/env ruby
#
# A simple function to create useful asserts
#
# 
# 
# 
#


#
# Exception raised when some swig binding test fails
#
class SwigRubyError < RuntimeError
end


#
# Asserts whether a and b are equal.
#
# scope - optional Binding where to run the code
# msg   - optional additional message to print
#
def swig_assert_equal( a, b, scope = nil, msg = nil )
  a = 'nil' if a == nil
  b = 'nil' if b == nil
  begin
    check = "#{a} == #{b}"
    if scope.kind_of? Binding
      ok = eval(check.to_s, scope)
    else
      ok = eval(check.to_s)
      if !msg
        msg = scope
        scope = nil
      end
    end
  rescue => e
    raise
  end

  unless ok
    valA = eval(a, scope)
    valB = eval(b, scope)
    raise SwigRubyError.new("FAILED EQUALITY: #{check} was #{valA} not #{valB}")
  end

  if $VERBOSE
    $stdout.puts "\tPASSED EQUALITY #{check} #{msg}"
  end

  return ok
rescue => e
  trace = e.backtrace[1..-1]
  $stderr.puts "#{trace[0,1]}: #{e}"
  if trace.size > 1
    $stderr.puts "\tfrom #{trace[1..-1].join("\n\t     ")}"
  end
  exit(1)
end


#
# Asserts whether an expression runs properly and is true
#
# scope - optional Binding where to run the code
# msg   - optional additional message to print
#
def swig_assert( expr, scope = nil, msg = nil )
  begin
    if scope.kind_of? Binding
      ok = eval(expr.to_s, scope)
    else
      ok = eval(expr.to_s)
      msg = scope if !msg
    end
  rescue
    raise
  end

  raise SwigRubyError.new("FAILED: #{expr.to_s} - #{msg}") unless ok

  if $VERBOSE
    $stdout.puts "\tPASSED #{expr} #{msg}"
  end
rescue => e
  trace = e.backtrace[1..-1]
  $stderr.puts "#{trace[0,1]}: #{e}"
  if trace.size > 1
    $stderr.puts "\tfrom #{trace[1..-1].join("\n\t     ")}"
  end
  exit(1)
end

#
# Asserts whether an expression runs properly
#
# scope - optional Binding where to run the code
# msg   - optional additional message to print
#
def swig_eval( expr, scope = nil, msg = nil )
  begin
    if scope.kind_of? Binding
      eval(expr.to_s, scope)
    else
      eval(expr.to_s)
      msg = scope if !msg
    end
  rescue => e
    raise SwigRubyError.new("Wrong assert: #{expr.to_s} - #{e}")
  end
  if $VERBOSE
    $stdout.puts "\tPASSED #{expr} #{msg}"
  end
rescue => e
  trace = e.backtrace[1..-1]
  $stderr.puts "#{trace[0,1]}: #{e}"
  if trace.size > 1
    $stderr.puts "\tfrom #{trace[1..-1].join("\n\t     ")}"
  end
  exit(1)
end


#
# Given a set of lines as text, runs each of them, asserting them.
# Lines that are of the form:
#     a == b  are run with swig_assert_equal
#             others are run with swig_eval.
#
# scope - optional Binding where to run the code
# msg   - optional additional message to print
#
def swig_assert_each_line( lines, scope = nil, msg = nil )
  lines.split("\n").each do |line|
    next if line.empty? or line =~ /^\s*#.*/
      if line =~ /^\s*([^\s]*)\s*==\s*(.*)\s*$/
        swig_assert_equal($1, $2, scope, msg)
      else
        swig_eval(line, scope, msg)
      end
  end
end
