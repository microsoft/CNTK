#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 4;
BEGIN { use_ok('typemap_namespace') }
require_ok('typemap_namespace');

is(typemap_namespace::test1("hello"), "hello", "test 1");
is(typemap_namespace::test2("hello"), "hello", "test 1");
