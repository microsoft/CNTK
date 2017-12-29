%module octave_dim

%include "std_vector.i"

namespace std {
    %template(IntVector) vector<int>;
}


%typemap(out) Matrix {
  $result = $1;
}

// Code below will not work. Kept for future reference.
// Reason: there is no octave_value(Array<octave_idx_type>) constructor
//%typemap(out) Array<octave_idx_type> {
//  $result = octave_value($1,true);
//}


%inline %{

class Foo45a {
public:
  std::vector<int> __dims__() const {
    std::vector<int> ret(2);
    ret[0] = 4;
    ret[1] = 5;
    return ret;
  }
};

// doubles are not converted to ints.
class Bar1 {
public:
  std::vector<double> __dims__() const {
    std::vector<double> ret(2);
    ret[0] = 4;
    ret[1] = 5;
    return ret;
  }
};

class Bar2 {
public:
  std::string __dims__() const {
    return "foo";
  }
};


class Foo4a {
public:
  std::vector<int> __dims__() const {
    std::vector<int> ret(1);
    ret[0] = 4;
    return ret;
  }
};

class Foo4b {
public:
  int __dims__() const {
    return 4;
  }
};

class Foo456a {
public:
  std::vector<int> __dims__() const {
    std::vector<int> ret(3);
    ret[0] = 4;
    ret[1] = 5;
    ret[2] = 6;
    return ret;
  }
};




class Foo {

};


class Baz1 {
public:
  Cell __dims__() const {
    Cell c(1,2);
    c(0) = 3;
    c(1) = 4;
    return c;
  }
};

class Baz2 {
public:
  Cell __dims__() const {
    Cell c(2,1);
    c(0) = 3;
    c(1) = 4;
    return c;
  }
};

class Baz3 {
public:
  Matrix __dims__() const {
    Matrix c(2,1);
    c(0) = 3;
    c(1) = 4;
    return c;
  }
};

class Baz4 {
public:
  Matrix __dims__() const {
    Matrix c(1,2);
    c(0) = 3;
    c(1) = 4;
    return c;
  }
};

class Baz5 {
public:
  Array<octave_idx_type> __dims__() const {
    Array<octave_idx_type> c(dim_vector(2,1));
    c(0) = 3;
    c(1) = 4;
    return c;
  }
};

class Baz6 {
public:
  Array<octave_idx_type> __dims__() const {
    Array<octave_idx_type> c(dim_vector(1,2));
    c(0) = 3;
    c(1) = 4;
    return c;
  }
};

// Code below will not work. Kept for future reference.
// Reason: there is no octave_value(dim_vector) constructor
// class Baz7 {
//public:
//  dim_vector __dims__() const {
//    octave_value v = dim_vector(3,4);
//    Array<int> a = v.int_vector_value();
//   if (error_state) return dim_vector(1,1);
//    int mysize = a.numel();
//    return dim_vector(3,4);
//  }
//};

%}
