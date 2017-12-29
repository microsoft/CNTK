%module li_std_carray

%include <std_carray.i>

%template(Vector3) std::carray<double, 3>;

%template(Matrix3) std::carray<std::carray<double, 3>, 3>;

