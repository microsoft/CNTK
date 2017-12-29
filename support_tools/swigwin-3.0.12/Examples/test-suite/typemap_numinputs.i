%module typemap_numinputs


%typemap(in, numinputs=1) (char *STR, int LEN)(int temp = 0)
{
  temp = 1;
  $2 = 0;
  $1 = 0;
}

%typemap(in) (int *OUTPUT)  (int temp = 0)
{
  temp = 2;
  $1 = &temp;
}

%typemap(argout) (int *OUTPUT)
{
  ++temp$argnum;
}

%typemap(argout, numinputs=1) (char *STR, int LEN)
{
  ++temp$argnum;
}

%typemap(in) int hello
{
  $1 = 0;
}

%inline %{
  int this_breaks(int hello, char *STR, int LEN, int *OUTPUT)
  {
    return LEN;
  }
%}
