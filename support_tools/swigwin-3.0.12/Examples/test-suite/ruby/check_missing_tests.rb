#!/usr/bin/env ruby
#
# This script allows you to compare the tests in the current directory
# (Ruby) against the tests of other languages to see which ones are missing
#
# 
# 
# 
#


ignore = ['ruby','std','typemaps']

curr = Dir.pwd.sub(/.*\//, '')

langs = Dir.glob('../*').select { |x| File.directory?("../#{x}") }
langs.map! { |x| x.sub(/^\.\.\/*/, '') }
langs -= ignore

# Add generic test directory, too
langs << ''

testsB = Dir.glob("*runme*").map { |x| x.sub(/\.\w+$/, '') }


all_tests = []

langs.each do |lang|
  testsA = Dir.glob("../#{lang}/*runme*")
  testsA.map! { |x| x.sub(/.*\/(\w+)\.\w+$/, '\1') }
  testsA.delete_if { |x| x =~ /~$/ } # ignore emacs backups

  diff = testsA - testsB

  unless diff.empty?
    puts '-'*70
    title = !lang.empty? ? "#{lang[0,1].upcase}#{lang[1..-1]}" : 'Generic'
    title = "Missing #{title} tests"
    puts title
    puts '='*title.size
    puts diff.join(', ')
    all_tests += diff 
  end

end


all_tests.uniq!

puts '-'*70
puts 'All missing tests'
puts '================='
puts all_tests.join(', ')
