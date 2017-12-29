exec("swigtest.start", -1);

b_set("hello");
checkequal(b_get(), "hello", "b_get()");

sa = new_A();
A_x_set(sa, 5);
checkequal(A_x_get(sa), 5, "A_x_get(sa)");

a_set(sa);
checkequal(A_x_get(a_get()), 5, "A_x_get(a)");

ap_set(sa);
A_x_set(sa, 14);
checkequal(A_x_get(ap_get()), 14, "A_x_get(ap)");
delete_A(sa);

sa2 = new_A();
cap_set(sa2);
A_x_set(sa2, 16);
checkequal(A_x_get(cap_get()), 16, "A_x_get(cap)");

checkequal(A_x_get(ar_get()), 5, "A_x_get(ar)");
ar_set(sa2);
checkequal(A_x_get(ar_get()), 16, "A_x_get(ar)");
delete_A(sa2);

x_set(11);
checkequal(x_get(), 11, "x_get()");

exec("swigtest.quit", -1);
