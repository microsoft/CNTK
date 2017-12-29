#!/usr/bin/env ruby
#
# Ruby's GC is somewhat broken in that it will mark some more stack space
# leading to the collection of local objects to be delayed.
# Thus, upon invokation, it sometimes you can wait up to several
# instructions to kick in.
# See: http://blade.nagaokaut.ac.jp/cgi-bin/scat.rb/ruby/ruby-core/7449
#
# 
# 
# 
#

require 'swig_assert'
require 'swig_gc'
require 'newobject2'

include Newobject2

GC.track_class = Foo
GC.stats if $VERBOSE
100.times { foo1 = makeFoo }
GC.stats if $VERBOSE
swig_assert( 'fooCount == 100', nil, "but is #{fooCount}" )
GC.start
swig_assert( 'fooCount <= 1', nil, "but is #{fooCount}" )

