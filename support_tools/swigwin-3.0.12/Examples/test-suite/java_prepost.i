%module java_prepost

// Test the pre, post attributes for javain typemaps

%include "std_vector.i"

%define VECTOR_DOUBLE_JAVAIN_POST
"      int count$javainput = (int)d$javainput.size();
      $javainput = new double[count$javainput];
      for (int i=0; i<count$javainput; ++i) {
        $javainput[i] = d$javainput.get(i);
      }"
%enddef

// pre and post in javain typemaps
//%typemap(jtype, nopgcpp=1) std::vector<double> &v "long" // could suppress pgcpp instead of using pgcppname, but not recommended
%typemap(jstype) std::vector<double> &v "double[]"
%typemap(javain, pre="    DoubleVector d$javainput = new DoubleVector();", post=VECTOR_DOUBLE_JAVAIN_POST, pgcppname="d$javainput") std::vector<double> &v
  "$javaclassname.getCPtr(d$javainput)"

%apply std::vector<double> & v { std::vector<double> & v2 }

// pre only in javain typemap
//%typemap(jtype, nopgcpp=1) std::vector<double> &vpre "long" // could suppress pgcpp instead of using pgcppname, but not recommended
%typemap(jstype) std::vector<double> &vpre "double[]"
%typemap(javain, pre="    DoubleVector d$javainput = new DoubleVector();\n    for (int i=0; i<$javainput.length; ++i) {\n      double d = $javainput[i];\n      d$javainput.add(d);\n    }", pgcppname="d$javainput") std::vector<double> &vpre
  "$javaclassname.getCPtr(d$javainput)"

// post only in javain typemap
%typemap(javain, post="      int size = (int)$javainput.size();\n      for (int i=0; i<size; ++i) {\n        $javainput.set(i, $javainput.get(i)/100);\n      }") std::vector<double> &vpost
  "$javaclassname.getCPtr($javainput)"

%inline %{
bool globalfunction(std::vector<double> & v) {
  v.push_back(0.0);
  v.push_back(1.1);
  v.push_back(2.2);
  return true;
}
struct PrePostTest {
  PrePostTest() {
  }
  PrePostTest(std::vector<double> & v) {
    v.push_back(3.3);
    v.push_back(4.4);
  }
  bool method(std::vector<double> & v) {
    v.push_back(5.5);
    v.push_back(6.6);
    return true;
  }
  static bool staticmethod(std::vector<double> & v) {
    v.push_back(7.7);
    v.push_back(8.8);
    return true;
  }
};

// Check pre and post only typemaps and that they coexist okay and that the generated code line spacing looks okay
bool globalfunction2(std::vector<double> & v, std::vector<double> &v2, std::vector<double> & vpre, std::vector<double> & vpost) {
  return true;
}
struct PrePost2 {
  PrePost2() {
  }
  PrePost2(std::vector<double> & v, std::vector<double> &v2, std::vector<double> & vpre, std::vector<double> & vpost) {
  }
  bool method(std::vector<double> & v, std::vector<double> &v2, std::vector<double> & vpre, std::vector<double> & vpost) {
    return true;
  }
  static bool staticmethod(std::vector<double> & v, std::vector<double> &v2, std::vector<double> & vpre, std::vector<double> & vpost) {
    return true;
  }
};
%}

%template(DoubleVector) std::vector<double>;


// Check pre post constructor helper deals with checked exceptions, InstantiationException is just a random checked exception
%typemap(javain, pre="    if ($javainput == null)\n      throw new InstantiationException(\"empty value!!\");", throws="InstantiationException") PrePostTest *
  "$javaclassname.getCPtr($javainput)"

%inline %{
struct PrePostThrows {
  PrePostThrows(PrePostTest *ppt, bool) {
  }
};
%}


