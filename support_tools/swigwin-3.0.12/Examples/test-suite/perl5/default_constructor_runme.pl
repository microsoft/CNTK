#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 20;
BEGIN { use_ok('default_constructor') }
require_ok('default_constructor');

isa_ok(eval { default_constructor::A->new() }, "default_constructor::A");
isa_ok(eval { default_constructor::AA->new() }, "default_constructor::AA");
is(    eval { default_constructor::B->new() }, undef, "private default constructor");
isa_ok(eval { default_constructor::B->new(0, 0) }, "default_constructor::B");
is(    eval { default_constructor::BB->new() }, undef, "inherited private default constructor");
is(    eval { default_constructor::C->new() }, undef, "protected default constructor");
isa_ok(eval { default_constructor::CC->new() }, "default_constructor::CC");
is(    eval { default_constructor::D->new() }, undef, "private constructor");
is(    eval { default_constructor::DD->new() }, undef, "inherited private constructor");
{ local $TODO = "default_constructor.i disagrees with our result";
is(    eval { default_constructor::AD->new() }, undef, "MI on A, D");
}
isa_ok(eval { default_constructor::E->new() }, "default_constructor::E");
isa_ok(eval { default_constructor::EE->new() }, "default_constructor::EE");
{ local $TODO = "default_constructor.i disagrees with our result";
is(    eval { default_constructor::EB->new() }, undef, "MI on E, B");
}
{ local $TODO = "destructor hiding seems off";
my $hit = 0;
eval {
	my $F = default_constructor::F->new();
	undef $F;
	$hit = 1;
};
ok(not($hit), "private destructor");
$hit = 0;
eval {
	my $G = default_constructor::G->new();
	undef $G;
	$hit = 1;
};
ok(not($hit), "protected destructor");
$hit = 0;
eval {
	my $G = default_constructor::GG->new();
	undef $G;
	$hit = 1;
};
ok(not($hit), "inherited protected destructor");
}
isa_ok(eval { default_constructor::HH->new(0, 0) }, "default_constructor::HH");
is(    eval { default_constructor::HH->new() }, undef, "templated protected constructor");

# TODO: sort out what needs to be tested from OSRSpatialReferenceShadow
