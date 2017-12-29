%module li_reference

%include "reference.i"

%inline %{
  double FrVal;
  double ToVal;
  void PDouble(double *REFERENCE, int t = 0)
    { ToVal = *REFERENCE; *REFERENCE = FrVal + t; }
  void RDouble(double &REFERENCE, int t = 0)
    { ToVal =  REFERENCE;  REFERENCE = FrVal + t; }
  void PFloat(float *REFERENCE, int t = 0)
    { ToVal = *REFERENCE; *REFERENCE = (float)(FrVal + t); }
  void RFloat(float &REFERENCE, int t = 0)
    { ToVal =  REFERENCE;  REFERENCE = (float)(FrVal + t); }
  void PInt(int *REFERENCE, int t = 0)
    { ToVal = *REFERENCE; *REFERENCE = (int)(FrVal + t); }
  void RInt(int &REFERENCE, int t = 0)
    { ToVal =  REFERENCE;  REFERENCE = (int)(FrVal + t); }
  void PShort(short *REFERENCE, int t = 0)
    { ToVal = *REFERENCE; *REFERENCE = (short)(FrVal + t); }
  void RShort(short &REFERENCE, int t = 0)
    { ToVal =  REFERENCE;  REFERENCE = (short)(FrVal + t); }
  void PLong(long *REFERENCE, int t = 0)
    { ToVal = *REFERENCE; *REFERENCE = (long)(FrVal + t); }
  void RLong(long &REFERENCE, int t = 0)
    { ToVal =  REFERENCE;  REFERENCE = (long)(FrVal + t); }
  void PUInt(unsigned int *REFERENCE, int t = 0)
    { ToVal = *REFERENCE; *REFERENCE = (unsigned int)(FrVal + t); }
  void RUInt(unsigned int &REFERENCE, int t = 0)
    { ToVal =  REFERENCE;  REFERENCE = (unsigned int)(FrVal + t); }
  void PUShort(unsigned short *REFERENCE, int t = 0)
    { ToVal = *REFERENCE; *REFERENCE = (unsigned short)(FrVal + t); }
  void RUShort(unsigned short &REFERENCE, int t = 0)
    { ToVal =  REFERENCE;  REFERENCE = (unsigned short)(FrVal + t); }
  void PULong(unsigned long *REFERENCE, int t = 0)
    { ToVal = *REFERENCE; *REFERENCE = (unsigned long)(FrVal + t); }
  void RULong(unsigned long &REFERENCE, int t = 0)
    { ToVal =  REFERENCE;  REFERENCE = (unsigned long)(FrVal + t); }
  void PUChar(unsigned char *REFERENCE, int t = 0)
    { ToVal = *REFERENCE; *REFERENCE = (unsigned char)(FrVal + t); }
  void RUChar(unsigned char &REFERENCE, int t = 0)
    { ToVal =  REFERENCE;  REFERENCE = (unsigned char)(FrVal + t); }
  void PChar(signed char *REFERENCE, int t = 0)
    { ToVal = *REFERENCE; *REFERENCE = (signed char)(FrVal + t); }
  void RChar(signed char &REFERENCE, int t = 0)
    { ToVal =  REFERENCE;  REFERENCE = (signed char)(FrVal + t); }
  void PBool(bool *REFERENCE, int t = 0)
    { ToVal = *REFERENCE; *REFERENCE = (FrVal + t) ? true : false; }
  void RBool(bool &REFERENCE, int t = 0)
    { ToVal =  REFERENCE;  REFERENCE = (FrVal + t) ? true : false; }
%}
