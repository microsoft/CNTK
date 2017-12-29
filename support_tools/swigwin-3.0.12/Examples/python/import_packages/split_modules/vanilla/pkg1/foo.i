%module(package="pkg1") foo
%{
static unsigned count(void)
{
  return 3;
}
%}

unsigned count(void);

