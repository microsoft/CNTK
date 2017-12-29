%module friends_template

#if defined(SWIGOCTAVE)
%warnfilter(SWIGWARN_IGNORE_OPERATOR_RSHIFT_MSG) operator>>;
#endif


%{
template <typename Type> class MyClass;

template <typename Type> int operator<<(double un, const MyClass <Type> & x) { return 0; }
template <typename Type> int funk_hidden(double is, MyClass <Type>  & x) { return 2; }

template <typename T> T template_friend_hidden(T t) { return t + 1; }
%}

%inline %{
template <typename Type> int operator>>(double is, MyClass <Type>  & x) { return 1; }
template <typename Type> int funk_seen(double is, MyClass <Type>  & x) { return 2; }
template <typename T> T template_friend_seen(T t1, T t2) { return t1 + t2; }
int friend_plain_seen(int i) { return i; }

template <class Type> class MyClass
{
  friend int operator<<  <Type>(double un, const MyClass <Type> & x);
  friend int operator>>  <Type>(double is, MyClass <Type> & x);
  friend int funk_hidden <Type>(double is, MyClass <Type> & x);
  friend int funk_seen   <Type>(double is, MyClass <Type> & x);
};

struct MyTemplate {
  template <typename T> friend T template_friend_hidden(T);
  template <typename T> friend T template_friend_seen(T, T);
  friend int friend_plain_seen(int i);
};

MyClass<int> makeMyClassInt() { return MyClass<int>(); }
%}

// Although the friends in MyClass are automatically instantiated via %template(MyClassDouble) MyClass<int>,
// the operator friends are not valid and hence %rename is needed.
%rename(OperatorInputDouble) operator>> <double>;
%rename(OperatorOutputDouble) operator<< <double>;
%template(MyClassDouble) MyClass<double>;

%template(TemplateFriendHiddenInt) template_friend_hidden<int>;
%template(TemplateFriendSeenInt) template_friend_seen<int>;

// These have no %template(XX) MyClass<int> to instantiate, but they can be instantiated separately...
%template(OperatorInputInt) operator>> <int>;
%template(OperatorFunkSeenInt) funk_seen <int>;
