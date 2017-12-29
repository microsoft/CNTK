exec("swigtest.start", -1);

// Negative values
checkequal(signed_char_identity(-1), -1, "signed_char_identity(-1)");
checkequal(signed_short_identity(-1), -1, "signed_short_identity(-1)");
checkequal(signed_int_identity(-1), -1, "signed_int_identity(-1)");
checkequal(signed_long_identity(-1), -1, "signed_long_identity(-1)");

// Overflow errors
ierr = execstr('signed_char_identity(2^8)', 'errcatch');
checkequal(ierr, 20007, 'signed_char_identity(2^8)');
ierr = execstr('signed_short_identity(2^16)', 'errcatch');
checkequal(ierr, 20007, 'signed_short_identity(2^16)');
ierr = execstr('signed_int_identity(2^32)', 'errcatch');
checkequal(ierr, 20007, 'signed_int_identity(2^32)');
ierr = execstr('signed_long_identity(2^64)', 'errcatch');
checkequal(ierr, 20007, 'signed_long_identity(2^64)');

// Value errors
ierr = execstr('signed_char_identity(100.2)', 'errcatch');
checkequal(ierr, 20009, 'signed_char_identity(100.2)');
ierr = execstr('signed_short_identity(100.2)', 'errcatch');
checkequal(ierr, 20009, 'signed_short_identity(100.2)');
ierr = execstr('signed_int_identity(100.2)', 'errcatch');
checkequal(ierr, 20009, 'signed_int_identity(100.2)');
ierr = execstr('signed_long_identity(100.2)', 'errcatch');
checkequal(ierr, 20009, 'signed_long_identity(100.2)');

exec("swigtest.quit", -1);
