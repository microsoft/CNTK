argcargvtest

largs={'hi','hola','hello'};
if (mainc(largs) != 3)
  error("bad main typemap");
endif

targs={'hi','hola'};
if (mainv(targs,1) != 'hola')
  error("bad main typemap");
endif

targs={'hi', 'hola'};
if (mainv(targs,1) != 'hola')
  error("bad main typemap");
endif

try
  error_flag = 0;
  mainv('hello',1);
  error_flag = 1;
catch
end_try_catch
if (error_flag)
  error("bad main typemap")
endif


initializeApp(largs);
