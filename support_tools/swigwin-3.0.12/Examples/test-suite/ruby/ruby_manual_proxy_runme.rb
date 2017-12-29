#!/usr/bin/env ruby
#
# The Subversion bindings use this manually written proxy class approach
# to the Ruby bindings. Note that in C the struct svn_fs_t is an
# opaque pointer and the Ruby FileSystem proxy class is hand written around it.
# This testcase tests this and the C close function and subsequent error
# handling.

require 'swig_assert'
require 'ruby_manual_proxy'

module Svn
  module Fs
    module_function
    def create(path)
      f = Ruby_manual_proxy::svn_fs_create(path)
      return f
    end

    FileSystem = SWIG::TYPE_p_svn_fs_t
    class FileSystem
      class << self
        def create(*args)
          Fs.create(*args)
        end
      end
      def path
        Ruby_manual_proxy::svn_fs_path(self)
      end
    end
  end
end

f = Svn::Fs::FileSystem.create("/tmp/myfile")
path = f.path
f.close
begin
  # regression in swig-3.0.8 meant ObjectPreviouslyDeleted error was thrown instead
  path = f.path
  raise RuntimeError.new("IOError (1) not thrown")
rescue IOError
end

file = nil
begin
  path = Ruby_manual_proxy::svn_fs_path(file)
  raise RuntimeError.new("IOError (2) not thrown")
rescue IOError
end
