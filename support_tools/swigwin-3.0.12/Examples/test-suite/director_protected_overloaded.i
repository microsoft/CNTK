%module(directors="1",dirprot="1") director_protected_overloaded

%director IDataObserver;
%director DerivedDataObserver;

// protected overloaded methods
%inline %{
  class IDataObserver
  {
    public:
      virtual ~IDataObserver(){}

    protected:
      virtual void notoverloaded() = 0;
      virtual void isoverloaded() = 0;
      virtual void isoverloaded(int i) = 0;
      virtual void isoverloaded(int i, double d) = 0;
  };
  class DerivedDataObserver : public IDataObserver {
  };
%}
