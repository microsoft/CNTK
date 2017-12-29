%module(directors="1") python_director

%feature("director");
%inline %{
  class IFactoryFuncs {
  public:
    IFactoryFuncs()           {}
    virtual ~IFactoryFuncs()  {}

    virtual PyObject * process(PyObject *pyobj) {
      return pyobj;
    }

    void process_again(const PyObject *& pyobj) {
    }
  };
%}

