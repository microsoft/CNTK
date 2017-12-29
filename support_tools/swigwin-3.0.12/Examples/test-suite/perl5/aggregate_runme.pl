#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 7;
BEGIN { use_ok('aggregate') }
require_ok('aggregate');

# adapted from ../java/aggregate_runme.java

# Confirm that move() returns correct results under normal use
is(aggregate::move($aggregate::UP), $aggregate::UP, "UP");

is(aggregate::move($aggregate::DOWN), $aggregate::DOWN, "DOWN");

is(aggregate::move($aggregate::LEFT), $aggregate::LEFT, "LEFT");

is(aggregate::move($aggregate::RIGHT), $aggregate::RIGHT, "RIGHT");

# Confirm that move() raises an exception when the contract is violated
eval { aggregate::move(0) };
like($@, qr/\bRuntimeError\b/);

