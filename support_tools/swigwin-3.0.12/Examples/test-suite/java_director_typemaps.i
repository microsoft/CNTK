%module(directors="1") java_director_typemaps

%feature("director", assumeoverride=1) Quux;

%include <typemaps.i>

%apply bool& OUTPUT {bool&};

%apply signed char& OUTPUT {signed char&};
%apply unsigned char& OUTPUT {unsigned char&};

%apply short& OUTPUT {short&};
%apply unsigned short& OUTPUT {unsigned short&};

%apply int& OUTPUT {int&};
%apply unsigned int& OUTPUT {unsigned int&};

%apply long& OUTPUT {long&};
%apply unsigned long& OUTPUT {unsigned long&};

%apply long long& OUTPUT {long long&};
// %apply unsigned long long& OUTPUT {unsigned long long&};

%apply float& OUTPUT {float&};
%apply double& OUTPUT {double&};

%apply bool& OUTPUT {bool& boolarg_output};

%apply signed char& OUTPUT {signed char& signed_chararg_output};
%apply unsigned char& OUTPUT {unsigned char& unsigned_chararg_output};

%apply short& OUTPUT {short& shortarg_output};
%apply unsigned short& OUTPUT {unsigned short& unsigned_shortarg_output};

%apply int& OUTPUT {int& intarg_output};
%apply unsigned int& OUTPUT {unsigned int& unsigned_intarg_output};

%apply long& OUTPUT {long& longarg_output};
%apply unsigned long& OUTPUT {unsigned long& unsigned_longarg_output};

%apply long long& OUTPUT {long long& long_longarg_output};
// %apply unsigned long long& OUTPUT {unsigned long long& unsigned_long_longarg_output};

%apply float& OUTPUT {float& floatarg_output};
%apply double& OUTPUT {double& doublearg_output};

%apply bool& INOUT {bool& boolarg_inout};

%apply signed char& INOUT {signed char& signed_chararg_inout};
%apply unsigned char& INOUT {unsigned char& unsigned_chararg_inout};

%apply short& INOUT {short& shortarg_inout};
%apply unsigned short& INOUT {unsigned short& unsigned_shortarg_inout};

%apply int& INOUT {int& intarg_inout};
%apply unsigned int& INOUT {unsigned int& unsigned_intarg_inout};

%apply long& INOUT {long& longarg_inout};
%apply unsigned long& INOUT {unsigned long& unsigned_longarg_inout};

%apply long long& INOUT {long long& long_longarg_inout};
// %apply unsigned long long& INOUT {unsigned long long& unsigned_long_longarg_inout};

%apply float& INOUT {float& floatarg_inout};
%apply double& INOUT {double& doublearg_inout};

%{
#include <stdexcept>
#define verify(ok) if (!(ok)) throw std::runtime_error(# ok);
%}
%inline %{

class Quux {
public:
  Quux() {}
  virtual ~Quux() {}

  virtual void director_method_bool_output(
    bool& boolarg_output,

    signed char& signed_chararg_output,
    unsigned char& unsigned_chararg_output,

    short& shortarg_output,
    unsigned short& unsigned_shortarg_output,

    int& intarg_output,
    unsigned int& unsigned_intarg_output,

    long& longarg_output,
    unsigned long& unsigned_longarg_output,

    long long& long_longarg_output,
    // unsigned long long& unsigned_long_longarg_output,

    float& floatarg_output,
    double& doublearg_output)
  {
    boolarg_output = false;

    signed_chararg_output = 50;
    unsigned_chararg_output = 50;

    shortarg_output = 50;
    unsigned_shortarg_output = 50;

    intarg_output = 50;
    unsigned_intarg_output = 50;

    longarg_output = 50;
    unsigned_longarg_output = 50;

    long_longarg_output = 50;
    // unsigned_long_longarg_output = 50;

    floatarg_output = 50;
    doublearg_output = 50;
  }

  virtual void director_method_bool_inout(
    bool& boolarg_inout,

    signed char& signed_chararg_inout,
    unsigned char& unsigned_chararg_inout,

    short& shortarg_inout,
    unsigned short& unsigned_shortarg_inout,

    int& intarg_inout,
    unsigned int& unsigned_intarg_inout,

    long& longarg_inout,
    unsigned long& unsigned_longarg_inout,

    long long& long_longarg_inout,
    // unsigned long long& unsigned_long_longarg_inout,

    float& floatarg_inout,
    double& doublearg_inout)
  {
    boolarg_inout = false;

    signed_chararg_inout = 50;
    unsigned_chararg_inout = 50;

    shortarg_inout = 50;
    unsigned_shortarg_inout = 50;

    intarg_inout = 50;
    unsigned_intarg_inout = 50;

    longarg_inout = 50;
    unsigned_longarg_inout = 50;

    long_longarg_inout = 50;
    // unsigned_long_longarg_inout = 50;

    floatarg_inout = 50;
    doublearg_inout = 50;
  }

  virtual void director_method_bool_nameless_args(
    bool& ,

    signed char& ,
    unsigned char& ,

    short& ,
    unsigned short& ,

    int& ,
    unsigned int& ,

    long& ,
    unsigned long& ,

    long long& ,
    // unsigned long long& ,

    float& ,
    double&)
  {
  }

  void etest() {
    bool boolarg_inout = false;

    signed char signed_chararg_inout = 150;
    unsigned char unsigned_chararg_inout = 150;

    short shortarg_inout = 150;
    unsigned short unsigned_shortarg_inout = 150;

    int intarg_inout = 150;
    unsigned int unsigned_intarg_inout = 150;

    long longarg_inout = 150;
    unsigned long unsigned_longarg_inout = 150;

    long long long_longarg_inout = 150;
    // unsigned long long unsigned_long_longarg_inout = 150;

    float floatarg_inout = 150;
    double doublearg_inout = 150;

    director_method_bool_output(
       boolarg_inout,

       signed_chararg_inout,
       unsigned_chararg_inout,

       shortarg_inout,
       unsigned_shortarg_inout,

       intarg_inout,
       unsigned_intarg_inout,

       longarg_inout,
       unsigned_longarg_inout,

       long_longarg_inout,
       // unsigned_long_longarg_inout,

       floatarg_inout,
       doublearg_inout);

    verify(boolarg_inout == true);
    verify(signed_chararg_inout == 1);
    verify(unsigned_chararg_inout == 2);

    verify(shortarg_inout == 3);
    verify(unsigned_shortarg_inout == 4);

    verify(intarg_inout == 5);
    verify(unsigned_intarg_inout == 6);

    verify(longarg_inout == 7);
    verify(unsigned_longarg_inout == 8);

    verify(long_longarg_inout == 9);
    // verify(unsigned_long_longarg_inout == 10);

    verify(floatarg_inout == 11);
    verify(doublearg_inout == 12);

    boolarg_inout = false;

    signed_chararg_inout = 101;
    unsigned_chararg_inout = 101;

    shortarg_inout = 101;
    unsigned_shortarg_inout = 101;

    intarg_inout = 101;
    unsigned_intarg_inout = 101;

    longarg_inout = 101;
    unsigned_longarg_inout = 101;

    long_longarg_inout = 101;
    // unsigned_long_longarg_inout = 101;

    floatarg_inout = 101;
    doublearg_inout = 101;

    director_method_bool_inout(
       boolarg_inout,

       signed_chararg_inout,
       unsigned_chararg_inout,

       shortarg_inout,
       unsigned_shortarg_inout,

       intarg_inout,
       unsigned_intarg_inout,

       longarg_inout,
       unsigned_longarg_inout,

       long_longarg_inout,
       // unsigned_long_longarg_inout,

       floatarg_inout,
       doublearg_inout);

    verify(boolarg_inout == false);
    verify(signed_chararg_inout == 11);
    verify(unsigned_chararg_inout == 12);

    verify(shortarg_inout == 13);
    verify(unsigned_shortarg_inout == 14);

    verify(intarg_inout == 15);
    verify(unsigned_intarg_inout == 16);

    verify(longarg_inout == 17);
    verify(unsigned_longarg_inout == 18);

    verify(long_longarg_inout == 19);
    // verify(unsigned_long_longarg_inout == 110);

    verify(floatarg_inout == 111);
    verify(doublearg_inout == 112);

    director_method_bool_nameless_args(
       boolarg_inout,

       signed_chararg_inout,
       unsigned_chararg_inout,

       shortarg_inout,
       unsigned_shortarg_inout,

       intarg_inout,
       unsigned_intarg_inout,

       longarg_inout,
       unsigned_longarg_inout,

       long_longarg_inout,
       // unsigned_long_longarg_inout,

       floatarg_inout,
       doublearg_inout);

    verify(boolarg_inout == true);
    verify(signed_chararg_inout == 12);
    verify(unsigned_chararg_inout == 13);

    verify(shortarg_inout == 14);
    verify(unsigned_shortarg_inout == 15);

    verify(intarg_inout == 16);
    verify(unsigned_intarg_inout == 17);

    verify(longarg_inout == 18);
    verify(unsigned_longarg_inout == 19);

    verify(long_longarg_inout == 20);
    // verify(unsigned_long_longarg_inout == 111);

    verify(floatarg_inout == 112);
    verify(doublearg_inout == 113);
  }
};
%}

%clear bool&;

%clear signed char&;
%clear unsigned char&;

%clear short&;
%clear unsigned short&;

%clear int&;
%clear unsigned int&;

%clear long&;
%clear unsigned long&;

%clear long long&;
// %clear unsigned long long&;

%clear float&;
%clear double&;
