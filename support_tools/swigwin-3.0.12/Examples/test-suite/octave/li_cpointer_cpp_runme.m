li_cpointer_cpp


p = new_intp();
intp_assign(p,3);

if (intp_value(p) != 3)
    error
endif

delete_intp(p);

