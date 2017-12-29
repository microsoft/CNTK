li_std_vector

iv = IntVector(4);
for i=0:3,
    iv(i) = i;
endfor
x = average(iv);

if (x != 1.5)
  error("average failed");
endif
