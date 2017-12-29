exec("swigtest.start", -1);

// These calls must fail
ierr = execstr('abstract_foo_meth(1)', 'errcatch');
if ierr == 0 then swigtesterror(); end

ierr = execstr('abstract_bar_meth(1)', 'errcatch');
if ierr == 0 then swigtesterror(); end

exec("swigtest.quit", -1);
