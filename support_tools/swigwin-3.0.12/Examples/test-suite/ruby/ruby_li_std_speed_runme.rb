#!/usr/bin/env ruby
#
# This is a simple speed benchmark suite for std containers,
# to verify their O(n) performance.
# It is not part of the standard tests.


require 'benchmark'
require 'mathn'
require 'ruby_li_std_speed'
include Ruby_li_std_speed


def benchmark(f, phigh, sequences)
  puts f
  print '%10s' % 'n'
  maxlen = sequences.max { |a,b| a.to_s.size <=> b.to_s.size }
  maxlen = maxlen.to_s.size - 12
  sequences.each { |s| print "%#{maxlen}s" % "#{s.to_s.sub(/.*::/,'')}" }
  puts
  o_perf = Array.new(sequences.size, 0)
  last_t = Array.new(sequences.size, nil)
  1.upto(phigh) do |p|
    n = 2**(p-1)
    print "%10d" % n
    sequences.each_with_index do |s, i|
      cont = s.new((0..n).to_a)
      Benchmark.benchmark('',0,"%#{maxlen-2}.6r") { |x|
        t = x.report { f.call(cont) }
        o_perf[i] += last_t[i] ? (t.real / last_t[i]) : t.real
        last_t[i] = t.real
      }
    end
    puts
  end

  print "  avg. O(n)"
  base = 1.0 / Math.log(2.0)
  sequences.each_with_index do |s, i|
    o_perf[i] /= phigh
    # o_perf[i] = 1 if o_perf[i] < 1
    o_perf[i]  = Math.log(o_perf[i]) * base
    print "%#{maxlen-1}.2f " % o_perf[i]
  end
  puts
end

def iterate(cont)
   it = cont.begin
   last = cont.end
   while it != last 
     it.next
   end
end


def erase(cont)
   it = cont.end
   # can't reuse begin since it might get invalidated
   while it != cont.begin
     it.previous
     # set returns None, so need to reobtain end
     it = cont.erase(it) || cont.end
   end
end

def insert(cont)
  size = cont.size
  size.upto((size<<1) - 1) { |x| cont.push(x) }
end

if $0 == __FILE__
  GC.disable
  sequences = [RbVector,RbDeque,RbSet,RbList,
               RbFloatVector,RbFloatDeque,RbFloatSet,RbFloatList]
  n = 17
  for f,phigh in [[method(:iterate),n], [method(:insert),n],
                  [method(:erase),n-4]]
    benchmark(f, phigh, sequences)
  end
end

