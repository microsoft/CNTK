# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

default_constructor

dc = default_constructor;

a = dc.new_A();
dc.delete_A(a);

aa = dc.new_AA();
dc.delete_AA(aa);

try
    b = dc.new_B();
    error("Whoa. new_BB created.")
catch
end_try_catch

try
    bb = dc.new_BB();
    error("Whoa. new_BB created.")
catch
end_try_catch

try
    c = dc.new_C();
    error("Whoa. new_C created.")
catch
end_try_catch

cc = dc.new_CC();
dc.delete_CC(cc);

try
    d = dc.new_D();
    error("Whoa. new_D created")
catch
end_try_catch

try
    dd = dc.new_DD();
    error("Whoa. new_DD created")
catch
end_try_catch

try
    ad = dc.new_AD();
    error("Whoa. new_AD created")
catch
end_try_catch

e = dc.new_E();
dc.delete_E(e);

ee = dc.new_EE();
dc.delete_EE(ee);

try
    eb = dc.new_EB();
    error("Whoa. new_EB created")
catch
end_try_catch

f = dc.new_F();

try
    del_f = dc.delete_F(f);
    error("Whoa. delete_F created")
catch
end_try_catch

dc.F_destroy(f);

g = dc.new_G();

try
    del_g = dc.delete_G(g);
    error("Whoa. delete_G created")
catch
end_try_catch

dc.G_destroy(g);

gg = dc.new_GG();
dc.delete_GG(gg);


hh = default_constructor.HH(1,1);


