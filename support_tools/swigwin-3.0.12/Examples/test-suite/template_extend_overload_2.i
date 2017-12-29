%module template_extend_overload_2

#ifdef SWIGLUA	// lua only has one numeric type, so some overloads shadow each other creating warnings
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) A;
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) AT;
%warnfilter(SWIGWARN_LANG_OVERLOAD_SHADOW) BT;
#endif

%inline %{

  struct A
  {
    A()
    {
    }
    
    A(int)
    {
    }

    int hi()
    {
      return 0;
    }
  };

  template <class T>
  struct AT
  {
    AT()
    {
    }
    
    AT(int)
    {
    }

    int hi()
    {
      return 0;
    }
  };

  template <class T>
  struct BT
  {
    BT()
    {
    }
    
    BT(int)
    {
    }

    int hi()
    {
      return 0;
    }
  };
  
%}


%extend A
{
  //
  // this works
  //

  int hi(int)
  {
    return 0;
  }

  A(double i)
  {
    A* a = new A();
    return a;
  }
}


%template(AT_double) AT<double>;
%extend AT<double>
{
  //
  // this doesn't work
  //

  int hi(int)
  {
    return 1;
  }
  
  AT<double>(double i)
  {
    AT<double>* a = new AT<double>();
    return a;
  }
}


%extend BT<double>
{
  //
  // this doesn't work either
  //

  int hi(int)
  {
    return 1;
  }
  
  BT<double>(double i)
  {
    BT<double>* a = new BT<double>();
    return a;
  }
}
%template(BT_double) BT<double>;

