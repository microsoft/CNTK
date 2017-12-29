#!/usr/bin/env ruby

require 'benchmark'
require 'li_std_speed'
include Li_std_speed

def benchmark(f, phigh, sequences)
  print f.class
  puts '%10s ' % 'n' + sequences.inject('') { |a,s| a << '%10s' % s.class }
  0.upto(phigh-1) do |p|
    n = 2**p
    print "%10d"%n
    $stdout.flush
    for s in sequences
      cont = s.new((0..n).to_a)
      Benchmark.benchmark { f.call(cont) }
    end
  end
end

def iterate(cont)
   # expected: O(n)
   # got: O(n**2) for set/list (vector/deque fine)
   it = cont.begin
   last = cont.end
   while it != last 
     it.next
   end
end


def erase(cont)
   # expected: O(n)
   # got: O(n**2) for vector/deque and O(n**3) for set/list
   it = cont.end
   # can't reuse begin since it might get invalidated
   while it != cont.begin
     it.previous
     # set returns None, so need to reobtain end
     it = cont.erase(it) or cont.end
   end
end

def insert(cont)
   it = cont.end
   size = cont.size
   if cont.kind_of? RbSet
       # swig stl missing hint version of insert for set
       # expected would be O(n) with iterator hint version
       # expected: O(n*log(n))
       # got: O(n**3*log(n))
     size.upto(size<<1) { |x| cont.insert(x) }
   else
     # expected: O(n)
     # got: O(n**3) for list (vector/deque fine)
     size.upto(size<<1) { |x| cont.push(x) }
   end
end

if $0 == __FILE__
  sequences = [RbVector,RbDeque,RbSet,RbList]
  for f,phigh in [[method(:iterate),15], [method(:insert),15],
                  [method(:erase),11]]
    benchmark(f, phigh, sequences)
  end
end
       
