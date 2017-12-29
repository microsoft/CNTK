# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

li_std_pair_extra

p = {1,2};
p1 = li_std_pair_extra.p_inout(p);
assert(all(cell2mat(p1)==[2,1]));
p2 = li_std_pair_extra.p_inoutd(p1);
assert(all(cell2mat(p2)==[1,2]));

d1 = li_std_pair_extra.d_inout(2);
assert(d1==4);

[i,d2] = li_std_pair_extra.d_inout2(2);
assert(all([i,d2]==[1,4]));

[i,p] = li_std_pair_extra.p_inout2(p);
assert(i==1&&all([cell2mat(p)]==[2,1]));
[p3,p4] = li_std_pair_extra.p_inout3(p1,p1);
assert(all(cell2mat(p3)==[2,1]));
assert(all(cell2mat(p4)==[2,1]));

psi = li_std_pair_extra.SIPair("hello",1);
assert(psi=={"hello",1});
pci = li_std_pair_extra.CIPair(complex(1,2),1);
assert(pci.first==complex(1,2)&&pci.second==1);


psi = li_std_pair_extra.SIPair("hi",1);
assert(psi.first=="hi"&&psi.second==1);

psii = li_std_pair_extra.SIIPair(psi,1);
assert(psii.first.first=="hi");
assert(psii.first.second==1);
assert(psii.second==1);

a = li_std_pair_extra.A();
b = li_std_pair_extra.B();

pab = li_std_pair_extra.ABPair(a,b);

pab.first = a;
pab.first.val = 2;

assert(pab.first.val == 2);

pci = li_std_pair_extra.CIntPair(1,0);
assert(pci.first==1&&pci.second==0);

a = li_std_pair_extra.A(5);
p1 = li_std_pair_extra.pairP1(1,a);
p2 = li_std_pair_extra.pairP2(a,1);
p3 = li_std_pair_extra.pairP3(a,a);

assert(a.val == li_std_pair_extra.p_identa(p1){2}.val);
  
p = li_std_pair_extra.IntPair(1,10);
assert(p.first==1&&p.second==10);
p.first = 1;
assert(p.first==1);

p = li_std_pair_extra.paircA1(1,a);
assert(p.first==1);
assert(swig_this(p.second)==swig_this(a));

p = li_std_pair_extra.paircA2(1,a);
assert(p.first==1);
assert(swig_this(p.second)==swig_this(a));
#pp = li_std_pair_extra.pairiiA(1,p); # conversion pb re const of pairA1/A2
pp = li_std_pair_extra.pairiiA(1,{1,A()});

