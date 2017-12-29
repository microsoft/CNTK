# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

director_string


function out=get_first(self)
  out = strcat(self.A.get_first()," world!");
end
function process_text(self,string)
  self.A.process_text(string);
  self.smem = "hello";
end
B=@(string) subclass(A(string),'get_first',@get_first,'process_text',@process_text);


b = B("hello");

b.get(0);
if (!strcmp(b.get_first(),"hello world!"))
  error(b.get_first())
endif

b.call_process_func();

if (!strcmp(b.smem,"hello"))
  error(b.smem)
endif

  
