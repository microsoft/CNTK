use strict;
use warnings;
use Test::More tests => 34;
BEGIN { use_ok('template_default_arg') }
require_ok('template_default_arg');

{
  my $helloInt = new template_default_arg::Hello_int();
  $helloInt->foo(0);
}
{
  my $x = new template_default_arg::X_int();
  is($x->meth(20.0, 200), 200, "X_int test 1");
  is($x->meth(20), 20, "X_int test 2");
  is($x->meth(), 0, "X_int test 3");
}

{
  my $y = new template_default_arg::Y_unsigned();
  is($y->meth(20.0, 200), 200, "Y_unsigned test 1");
  is($y->meth(20), 20, "Y_unsigned test 2");
  is($y->meth(), 0, "Y_unsigned test 3");
}

{
  my $x = new template_default_arg::X_longlong();
  $x = new template_default_arg::X_longlong(20.0);
  $x = new template_default_arg::X_longlong(20.0, 200);
}
{
  my $x = new template_default_arg::X_int();
  $x = new template_default_arg::X_int(20.0);
  $x = new template_default_arg::X_int(20.0, 200);
}
{
  my $x = new template_default_arg::X_hello_unsigned();
  $x = new template_default_arg::X_hello_unsigned(20.0);
  $x = new template_default_arg::X_hello_unsigned(20.0, new template_default_arg::Hello_int());
}
{
  my $y = new template_default_arg::Y_hello_unsigned();
  $y->meth(20.0, new template_default_arg::Hello_int());
  $y->meth(new template_default_arg::Hello_int());
  $y->meth();
}

{
  my $fz = new template_default_arg::Foo_Z_8();
  my $x = new template_default_arg::X_Foo_Z_8();
  my $fzc = $x->meth($fz);
}

# Templated functions
{
  # plain function: int ott(Foo<int>)
  is(template_default_arg::ott(new template_default_arg::Foo_int()), 30, "ott test 1");

  # %template(ott) ott<int, int>;
  is(template_default_arg::ott(), 10, "ott test 2");
  is(template_default_arg::ott(1), 10, "ott test 3");
  is(template_default_arg::ott(1, 1), 10, "ott test 4");

  is(template_default_arg::ott("hi"), 20, "ott test 5");
  is(template_default_arg::ott("hi", 1), 20, "ott test 6");
  is(template_default_arg::ott("hi", 1, 1), 20,"ott test 7");

  # %template(ott) ott<const char *>;
  is(template_default_arg::ottstring(new template_default_arg::Hello_int(), "hi"), 40, "ott test 8");

  is(template_default_arg::ottstring(new template_default_arg::Hello_int()), 40, "ott test 9");

  # %template(ott) ott<int>;
  is(template_default_arg::ottint(new template_default_arg::Hello_int(), 1), 50, "ott test 10");

  is(template_default_arg::ottint(new template_default_arg::Hello_int()), 50, "ott test 11");

  # %template(ott) ott<double>;
  is(template_default_arg::ott(new template_default_arg::Hello_int(), 1.0), 60, "ott test 12");

  is(template_default_arg::ott(new template_default_arg::Hello_int()), 60, "ott test 13");
}

# Above test in namespaces
{
  # plain function: int nsott(Foo<int>)
  is(template_default_arg::nsott(new template_default_arg::Foo_int()), 130, "nsott test 1");

  # %template(nsott) nsott<int, int>;
  is(template_default_arg::nsott(), 110, "nsott test 2");
  is(template_default_arg::nsott(1), 110, "nsott test 3");
  is(template_default_arg::nsott(1, 1), 110,  "nsott test 4");

  is(template_default_arg::nsott("hi"), 120, "nsott test 5");
  is(template_default_arg::nsott("hi", 1), 120, "nsott test 6");
  is(template_default_arg::nsott("hi", 1, 1), 120, "nsott test 7");

  # %template(nsott) nsott<const char *>;
  is(template_default_arg::nsottstring(new template_default_arg::Hello_int(), "hi"), 140, "nsott test 8");

  is(template_default_arg::nsottstring(new template_default_arg::Hello_int()), 140, "nsott test 9");

  # %template(nsott) nsott<int>;
  is(template_default_arg::nsottint(new template_default_arg::Hello_int(), 1), 150, "nsott test 10");

  is(template_default_arg::nsottint(new template_default_arg::Hello_int()), 150, "nsott test 11");

  # %template(nsott) nsott<double>;
  is(template_default_arg::nsott(new template_default_arg::Hello_int(), 1.0), 160, "nsott test 12");

  is(template_default_arg::nsott(new template_default_arg::Hello_int()), 160, "nsott test 13");
}
