li_std_carray


v3 = Vector3();
for i=0:len(v3),
    v3(i) = i;
endfor

i = 0;
for d in v3,
  if (d != i)
    error
  endif
  i = i + 1;
endfor


m3 = Matrix3();

for i=0:len(m3),
  v3 = m3(i);
  for j=0:len(v3),
    v3(j) = i + j;
  endfor
endfor

i = 0;
for v3 in m3,
  j = 0;
  for d in v3,
    if (d != i + j)
      error
    endif
    j = j + 1;
  endfor
  i = i + 1
endfor

for i=0:len(m3),
  for j=0:len(m3),
    if (m3(i,j) != i + j)
      error
    endif
  endfor
endfor

da = Vector3([1,2,3]);
ma = Matrix3({[1,2,3],[4,5,6],[7,8,9]});




