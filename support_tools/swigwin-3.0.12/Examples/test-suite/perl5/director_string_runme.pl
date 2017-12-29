use strict;
use warnings;
use Test::More tests => 5;
BEGIN { use_ok 'director_string' }
require_ok 'director_string';

{
  package B;
  use base 'director_string::A';
  our $in_first = 0;
  sub get_first { my($self) = @_;
    die "SUPER RESOLVE BAD" if $in_first;
    local $in_first = 1;
    return $self->SUPER::get_first() . " world!";
  }
  our $in_process_text = 0;
  sub process_text { my($self, $string) = @_;
    die "SUPER RESOLVE BAD" if $in_process_text;
    local $in_process_text = 1;
    $self->SUPER::process_text($string);
    $self->{'smem'} = "hello";
  }
}

my $b = B->new("hello");
isa_ok $b, 'B';

$b->get(0);

is $b->get_first(),  "hello world!";

$b->call_process_func();

is $b->{'smem'}, "hello";
