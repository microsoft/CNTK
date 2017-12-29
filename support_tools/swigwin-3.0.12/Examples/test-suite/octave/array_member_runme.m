array_member

f = Foo();
f.data = cvar.global_data;

for i=0:7,
    if (get_value(f.data,i) != get_value(cvar.global_data,i))
      error("Bad array assignment");
    endif
endfor

for i=0:7,
    set_value(f.data,i,-i);
endfor

cvar.global_data = f.data;

for i=0:7,
  if (get_value(f.data,i) != get_value(cvar.global_data,i))
    error("Bad array assignment")
  endif
endfor



