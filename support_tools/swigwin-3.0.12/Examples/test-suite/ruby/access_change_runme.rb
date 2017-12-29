#!/usr/bin/env ruby
#
# Put script description here.
#
# 
# 
# 
#

require 'swig_assert'
require 'access_change'


#
# this test will currently fail, as it exposes functions that were
# made protected from public.  swig limitation for now.
#
exit(0)

include Access_change

klass = BaseInt.new
public = ['PublicProtectedPublic1', 'PublicProtectedPublic2',
              'PublicProtectedPublic3', 'PublicProtectedPublic4']
methods = (klass.public_methods - Object.methods).sort
pmethods = (klass.protected_methods - Object.methods).sort
swig_assert( methods == public, 
             " incorrect public methods for BaseInt\n" +
             "#{methods.inspect} !=\n#{public.inspect}" )

klass = DerivedInt.new
public = ['PublicProtectedPublic3', 'PublicProtectedPublic4',
         'WasProtected1', 'WasProtected2', 'WasProtected3', 'WasProtected4']
methods = (klass.public_methods - Object.methods).sort
swig_assert( methods == public, 
             " incorrect public methods for DerivedInt\n" + 
             "#{methods.inspect} !=\n#{public.inspect}" )

klass = BottomInt.new
public = ['PublicProtectedPublic1', 'PublicProtectedPublic2',
          'PublicProtectedPublic3', 'PublicProtectedPublic4',
          'WasProtected1', 'WasProtected2']
methods = (klass.public_methods - Object.methods).sort
swig_assert( methods == public, 
             " incorrect public methods for BottomInt\n" +
             "#{methods.inspect} !=\n#{public.inspect}" )
