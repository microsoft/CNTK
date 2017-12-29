#!/usr/bin/env ruby
#
#
# VERY nice function from Robert Klemme to check memory leaks
# and check on what GC has collected since last call.
#
# Usage can be:
#
#     require 'swig_gc'
#
#     GC.stats
#     # do some stuff..
#     GC.start  # collect and report stats
#     # do some more...
#     GC.stats  # just report stats
#
# or:
#
#     require 'swig_gc'
#
#     GC.track_class = String  # track just String classes
#     GC.stats
#     # do some stuff..
#     GC.start  # collect and report stats
#     # do some more...
#     GC.stats  # just report stats
#
# 
# 
# 
#

module GC

  class << self
      
    attr          :last_stat
    attr_accessor :track_class

    alias :_start :start
    
    def start
      _start
      stats if $VERBOSE
    end
    
    def stats
      stats = Hash.new(0)
      ObjectSpace.each_object {|o| stats[o.class] += 1}
      
      if track_class
        v = stats[track_class]
        printf "\t%-30s  %10d", track_class.to_s, v
        if last_stat
          printf " | delta %10d", (v - last_stat[track_class])
        end
        puts
      else
        stats.sort {|(k1,v1),(k2,v2)| v2 <=> v1}.each do |k,v|
          printf "\t%-30s  %10d", k, v
          printf " | delta %10d", (v - last_stat[k]) if last_stat
          puts
        end
      end
        
      last_stat = stats
    end
  end
  
end
