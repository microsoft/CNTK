director_detect 

global MyBar=@(val=2) subclass(director_detect.Bar(),'val',val,@get_value,@get_class,@just_do_it,@clone);

function val=get_value(self)
    self.val = self.val + 1;
    val = self.val;
end
function ptr=get_class(self)
  global director_detect;
  self.val = self.val + 1;
  ptr=director_detect.A();
end
function just_do_it(self)
  self.val = self.val + 1;
end
function ptr=clone(self)
  global MyBar;
  ptr=MyBar(self.val);
end

b = MyBar();

f = b.baseclass();

v = f.get_value();
a = f.get_class();
f.just_do_it();

c = b.clone();
vc = c.get_value();

if ((v != 3) || (b.val != 5) || (vc != 6))
  error("Bad virtual detection")
endif

