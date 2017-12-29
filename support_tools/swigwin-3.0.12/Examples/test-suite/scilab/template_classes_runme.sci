exec("swigtest.start", -1);

ri = new_RectangleInt();
pi = RectangleInt_getPoint(ri);
x = PointInt_getX(pi);
delete_RectangleInt(ri);

exec("swigtest.quit", -1);
