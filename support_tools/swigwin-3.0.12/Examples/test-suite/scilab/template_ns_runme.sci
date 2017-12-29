exec("swigtest.start", -1);

p1 = new_pairii(2, 3);
p2 = new_pairii(p1);

checkequal(pairii_first_get(p2), 2, "pairii_first(p2) test fails.");
checkequal(pairii_second_get(p2), 3, "pairii_second(p2) test fails.");

p3 = new_pairdd(0.5, 2.5);
p4 = new_pairdd(p3);

checkequal(pairdd_first_get(p4), 0.5, "pairdd_first(p4) test fails.");
checkequal(pairdd_second_get(p4), 2.5, "pairdd_second(p4) test fails.");

exec("swigtest.quit", -1);
