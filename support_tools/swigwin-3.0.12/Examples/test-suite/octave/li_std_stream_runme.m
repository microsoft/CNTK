li_std_stream

a = A();

o = ostringstream();

o << a << " " << 2345 << " " << 1.435;


if (o.str() !=  "A class 2345 1.435")
  error
endif

