exec("swigtest.start", -1);


f1 = new_Foo1(3);
f11 = new_Foo1(f1);

checkequal(Foo1_x_get(f1), Foo1_x_get(f11), "Foo1_x_get(f1) <> Foo1_x_get(f11)");

delete_Foo1(f1);
delete_Foo1(f11);

f8 = new_Foo8();
try
  f81 = new_Foo8(f8);
  swigtesterror("Foo(f8) called.");
catch
end

bi = new_Bari(5);
bc = new_Bari(bi);

checkequal(Bari_x_get(bi), Bari_x_get(bc), "Bar_x_get(bi) <> Bar_x_get(bc)");

delete_Bari(bi);
delete_Bari(bc);

bd = new_Bard(5);
try
  bc = Bard(bd);
  swigtesterror("Bard(bd) called.");
catch
end


exec("swigtest.quit", -1);

