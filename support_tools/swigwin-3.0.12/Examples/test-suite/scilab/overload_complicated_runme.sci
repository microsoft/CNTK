exec("swigtest.start", -1);

NULL = SWIG_ptr(0);
p = new_Pop(NULL);
p = new_Pop(NULL, %T);

checkequal(Pop_hip(p, %T), 701, "Pop_hip(%T)");
checkequal(Pop_hip(p, NULL), 702, "Pop_hip(NULL)");

checkequal(Pop_hop(p, %T), 801, "Pop_hop(%T)");
checkequal(Pop_hop(p, NULL), 805, "Pop_hop(NULL)");

checkequal(Pop_pop(p, %T), 901, "Pop_pop(%T)");
checkequal(Pop_pop(p, NULL), 904, "Pop_pop(NULL)");
checkequal(Pop_pop(p), 905, "Pop_pop()");

checkequal(Pop_bop(p, NULL), 1001, "Pop_bop(NULL)");

checkequal(Pop_bip(p, NULL), 2002, "Pop_bip(%T)");

checkequal(muzak(%T), 3001, "muzak(%T)");
checkequal(muzak(NULL), 3002, "muzak(%T)");

exec("swigtest.quit", -1);

