# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

std_containers


cube = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};

icube = std_containers.cident(cube);
for i=len(cube),
  if (cube{i} != icube{i})
    error("bad cident")
  endif
endfor


p = {1,2};
if (p != std_containers.pident(p))
  error("bad pident")
endif

v = {1,2,3,4,5,6};
iv = std_containers.vident(v);
for i=0:len(v),
  if (v{i} != iv{i})
    error("bad vident")
  endif
endfor


iv = std_containers.videntu(v);
for i=0:len(v),
  if (v{i} != iv{i})
    error("bad videntu")
  endif
endfor


vu = std_containers.vector_ui(v);
if (vu{2} != std_containers.videntu(vu){2})
  error("bad videntu")
endif
  

if (v(0:3)(1) != vu(0:3)(1))
  error("bad getslice")
endif
  

m = {[1,2,3],[2,3],[3,4]};
im = std_containers.midenti(m);

for i=0:len(m),
  for j=0:len(m(i)),
    if (m(i)(j) != im(i)(j))
      error("bad getslice")
    endif
  endfor
endfor

m = {[1,0,1],[1,1],[1,1]};
im = std_containers.midentb(m);
for i=0:len(m),
  for j in range(0,len(m[i])):
    if (m(i)(j) != im(i)(j))
      error("bad getslice")
    endif
  endfor
endfor


mi = std_containers.imatrix(m);
mc = std_containers.cmatrix(m);
if (mi(0)(1) != mc(0)(1))
  error("bad matrix")
endif


map ={};
map{'hello'} = 1;
map{'hi'} = 2;
map{'3'} = 2;

imap = std_containers.mapident(map);
for k in map,
  if (map(k) != imap(k))
    error("bad map")
  endif
endfor


mapc ={};
c1 = std_containers.C();
c2 = std_containers.C();
mapc(1) = swig_this(c1);
mapc(2) = c2;

std_containers.mapidentc(mapc);


vi = std_containers.vector_i([2,2,3,4]);


v = [1,2];
v1 = std_containers.v_inout(vi);
vi(1), v1(1);

[v1,v2] = std_containers.v_inout2(v1,v2);

a1 = std_containers.A(1);
a2 = std_containers.A(2);

p1 = {1,a1};
p2 = {2,a2};
v = (p1,p2);
v2= std_containers.pia_vident(v);

v2(0)(1).a;
v2(1)(1).a;

v3 = std_containers.vector_piA(v2);

v3(0)(1).a;
v3(1)(1).a;

s = std_containers.set_i();
s.append(1);
s.append(2);
s.append(3);
j=1;
for i in s,
  if (i != j)
    error
  endif
  j = j + 1;
endfor


