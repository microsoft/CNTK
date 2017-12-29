%module (directors="1") csharp_prepost

// Test the pre, post, terminate and cshin attributes for csin typemaps

%include "std_vector.i"

%define VECTOR_DOUBLE_CSIN_POST
"      int count$csinput = d$csinput.Count;
      $csinput = new double[count$csinput];
      for (int i=0; i<count$csinput; ++i) {
        $csinput[i] = d$csinput[i];
      }"
%enddef

// pre and post in csin typemaps
%typemap(cstype) std::vector<double> &v "out double[]"
%typemap(csin, pre="    DoubleVector d$csinput = new DoubleVector();", post=VECTOR_DOUBLE_CSIN_POST, cshin="out $csinput") std::vector<double> &v
  "$csclassname.getCPtr(d$csinput)"

%apply std::vector<double> & v { std::vector<double> & v2 }

// pre only in csin typemap
%typemap(cstype) std::vector<double> &vpre "ref double[]"
%typemap(csin, pre="    DoubleVector d$csinput = new DoubleVector();\n    foreach (double d in $csinput) {\n      d$csinput.Add(d);\n    }", cshin="ref $csinput") std::vector<double> &vpre
  "$csclassname.getCPtr(d$csinput)"

// post only in csin typemap
%typemap(csin, post="      int size = $csinput.Count;\n"
                    "      for (int i=0; i<size; ++i) {\n"
                    "        $csinput[i] /= 100;\n"
                    "      }") std::vector<double> &vpost
  "$csclassname.getCPtr($csinput)"

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
  virtual ~PrePost2() {
  }
  PrePost2(std::vector<double> & v, std::vector<double> &v2, std::vector<double> & vpre, std::vector<double> & vpost) {
  }
  virtual bool method(std::vector<double> & v, std::vector<double> &v2, std::vector<double> & vpre, std::vector<double> & vpost) {
    return true;
  }
  static bool staticmethod(std::vector<double> & v, std::vector<double> &v2, std::vector<double> & vpre, std::vector<double> & vpost) {
    return true;
  }
};
%}

// Check csdirectorin pre and post attributes
// ref param
%typemap(csdirectorin,
   pre="    DoubleVector d$iminput = new DoubleVector($iminput, false);\n"
       "    int count$iminput = d$iminput.Count;\n"
       "    double[] v$iminput = new double[count$iminput];\n"
       "    for (int i=0; i<count$iminput; ++i) {\n"
       "      v$iminput[i] = d$iminput[i];\n"
       "    }\n",
   post="      foreach (double d in v$iminput) {\n"
        "        d$iminput.Add(d);\n"
        "      }\n"
  ) std::vector<double> &vpre
  "ref v$iminput"
// post only in csdirectorin typemap
%typemap(csdirectorin, post="      DoubleVector d$iminput = new DoubleVector($iminput, false);\n"
                            "      int size = d$iminput.Count;\n"
                            "      for (int i=0; i<size; ++i) {\n"
                            "        d$iminput[i] /= 100;\n"
                            "      }") std::vector<double> &vpost
  "new $csclassname($iminput, false)"

%feature("director") PrePost3;
%inline %{
struct PrePost3 {
  PrePost3() {
  }
  virtual ~PrePost3(){}
  virtual void method(std::vector<double> & vpre, std::vector<double> & vpost) {}
  virtual int methodint(std::vector<double> & vpre, std::vector<double> & vpost) { return 0; }
};
%}


%template(DoubleVector) std::vector<double>;

// Check attributes in the typemaps
%typemap(cstype, inattributes="[CustomInt]") int val "int"
%typemap(csin, pre="    int tmp_$csinput = $csinput * 100;") int val "tmp_$csinput"
%typemap(imtype, out="global::System.IntPtr/*overridden*/", outattributes="[CustomIntPtr]") CsinAttributes * "global::System.Runtime.InteropServices.HandleRef/*overridden*/"

%inline %{
class CsinAttributes {
  int m_val;
public:
  CsinAttributes(int val) : m_val(val) {}
  int getVal() { return m_val; }
};
%}



// test Date marshalling with pre post and terminate typemap attributes (Documented in CSharp.html)
%typemap(cstype) const CDate& "System.DateTime"
%typemap(csin, 
         pre="    CDate temp$csinput = new CDate($csinput.Year, $csinput.Month, $csinput.Day);"
        ) const CDate &
         "$csclassname.getCPtr(temp$csinput)"

%typemap(cstype) CDate& "out System.DateTime"
%typemap(csin, 
         pre="    CDate temp$csinput = new CDate();", 
         post="      $csinput = new System.DateTime(temp$csinput.getYear(),"
              " temp$csinput.getMonth(), temp$csinput.getDay(), 0, 0, 0);", 
         cshin="out $csinput"
        ) CDate &
         "$csclassname.getCPtr(temp$csinput)"


%inline %{
class CDate {
public:
  CDate();
  CDate(int year, int month, int day);
  int getYear();
  int getMonth();
  int getDay();
private:
  int m_year;
  int m_month;
  int m_day;
};
struct Action {
  int doSomething(const CDate &dateIn, CDate &dateOut);
  Action(const CDate &dateIn, CDate& dateOut);
};
%}

%{
Action::Action(const CDate &dateIn, CDate& dateOut) {dateOut = dateIn;}
int Action::doSomething(const CDate &dateIn, CDate &dateOut) { dateOut = dateIn; return 0; }
CDate::CDate() : m_year(0), m_month(0), m_day(0) {}
CDate::CDate(int year, int month, int day) : m_year(year), m_month(month), m_day(day) {}
int CDate::getYear() { return m_year; }
int CDate::getMonth() { return m_month; }
int CDate::getDay() { return m_day; }
%}

%typemap(cstype, out="System.DateTime") CDate * "ref System.DateTime"

%typemap(csin,
         pre="    CDate temp$csinput = new CDate($csinput.Year, $csinput.Month, $csinput.Day);",
         post="      $csinput = new System.DateTime(temp$csinput.getYear(),"
              " temp$csinput.getMonth(), temp$csinput.getDay(), 0, 0, 0);", 
         cshin="ref $csinput"
        ) CDate *
         "$csclassname.getCPtr(temp$csinput)"

%inline %{
void addYears(CDate *pDate, int years) {
  *pDate = CDate(pDate->getYear() + years, pDate->getMonth(), pDate->getDay());
}
%}

%typemap(csin,
         pre="    using (CDate temp$csinput = new CDate($csinput.Year, $csinput.Month, $csinput.Day)) {",
         post="      $csinput = new System.DateTime(temp$csinput.getYear(),"
              " temp$csinput.getMonth(), temp$csinput.getDay(), 0, 0, 0);", 
         terminator="    } // terminate temp$csinput using block",
         cshin="ref $csinput"
        ) CDate *
         "$csclassname.getCPtr(temp$csinput)"

%inline %{
void subtractYears(CDate *pDate, int years) {
  *pDate = CDate(pDate->getYear() - years, pDate->getMonth(), pDate->getDay());
}
%}

%typemap(csvarin, excode=SWIGEXCODE2) CDate * %{
    /* csvarin typemap code */
    set {
      CDate temp$csinput = new CDate($csinput.Year, $csinput.Month, $csinput.Day);
      $imcall;$excode
    } %}

%typemap(csvarout, excode=SWIGEXCODE2) CDate * %{
    /* csvarout typemap code */
    get {
      global::System.IntPtr cPtr = $imcall;
      CDate tempDate = (cPtr == global::System.IntPtr.Zero) ? null : new CDate(cPtr, $owner);$excode
      return new System.DateTime(tempDate.getYear(), tempDate.getMonth(), tempDate.getDay(),
                                 0, 0, 0);
    } %}

%inline %{
CDate ImportantDate = CDate(1999, 12, 31);
struct Person {
  CDate Birthday;
};
%}

