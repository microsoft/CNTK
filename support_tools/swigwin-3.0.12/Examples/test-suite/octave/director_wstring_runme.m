# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

director_wstring


B=@(string) subclass(A(string),\
		     'get_first',A.get_first(self) + " world!",\
		     'process_text',@(self) self.smem = u"hello"\
		     );

b = B("hello");

b.get(0);
if (!strcmp(b.get_first(),"hello world!"))
  error(b.get_first())
endif

b.call_process_func();

if (!strcmp(b.smem,"hello"))
  error(smem)
endif
  
