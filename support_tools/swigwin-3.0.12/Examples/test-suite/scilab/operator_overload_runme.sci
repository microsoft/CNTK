exec("swigtest.start", -1);

function checktrue(value, msg)
  checkequal(value, %T, msg)
endfunction

a = new_Op();
b = new_Op(5);
c = new_Op(b);
d = new_Op(2);
dd = new_Op();

// Assignment operator
Op_Equal(dd, d);

// Comparison operator
checktrue(Op_NotEqual(a, b), "Op_NotEqual(a, b)");
checktrue(Op_EqualEqual(b, c), "Op_EqualEqual(b, c)");
checktrue(Op_NotEqual(a, d), "Op_NotEqual(a, d)");
checktrue(Op_EqualEqual(d, dd), "Op_EqualEqual(d, dd)");

checktrue(Op_LessThan(a, b), "Op_LessThan(a, b)");
checktrue(Op_LessThanEqual(a, b), "Op_LessThanEqual(a, b)");
checktrue(Op_LessThanEqual(b, c), "Op_LessThanEqual(b, c)");
checktrue(Op_GreaterThanEqual(b, c), "Op_GreaterThanEqual(b, c)");
checktrue(Op_GreaterThan(b, d), "Op_GreaterThan(b, d)");
checktrue(Op_GreaterThanEqual(b, d), "Op_GreaterThanEqual(b, d)");

delete_Op(a);
delete_Op(b);
delete_Op(c);
delete_Op(d);
delete_Op(dd);

f = new_Op(1);
g = new_Op(1);

expop = new_Op();

op = Op_Plus(f, g);
Op_i_set(expop, 2);
checktrue(Op_EqualEqual(op, expop), "Op_Plus(f, g) <> Op(2)");
delete_Op(op);

op = Op_Minus(f, g);
Op_i_set(expop, 0);
checktrue(Op_EqualEqual(op, expop), "Op_Minus(f, g) <> Op(0)");
delete_Op(op);

op = Op_Multiply(f, g);
Op_i_set(expop, 1);
checktrue(Op_EqualEqual(op, expop), "Op_Multiply(f, g) <> Op(1)");
delete_Op(op);

op = Op_Divide(f, g);
Op_i_set(expop, 1);
checktrue(Op_EqualEqual(op, expop), "Op_Divide(f, g) <> Op(1)");
delete_Op(op);

// Unary operator
op = Op_PlusPlusPrefix(new_Op(3));
Op_i_set(expop, 4);
checktrue(Op_EqualEqual(op, expop), "Op_PlusPlusPrefix(op) <> Op(4)");

// Square bracket operator
checkequal(Op_IndexIntoConst(op, uint32(0)), 4, "Op_IndexIntoConst(op, 0) <> 4");
checkequal(Op_IndexIntoConst(op, uint32(1)), 0, "Op_IndexIntoConst(op, 1) <> 0");

// Functor
i = new_Op(3);
checkequal(Op_Functor(i), 3, "Op_Functor(i)");
checkequal(Op_Functor(i, 1), 4, "Op_Functor(i, 1)");

delete_Op(f);
delete_Op(g);

delete_Op(i);

delete_Op(expop);

exec("swigtest.quit", -1);

