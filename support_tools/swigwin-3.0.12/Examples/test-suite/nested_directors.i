%module(directors="1", allprotected="1") nested_directors

%warnfilter(SWIGWARN_PARSE_NAMED_NESTED_CLASS) NN::Base::Nest;
%warnfilter(SWIGWARN_PARSE_NAMED_NESTED_CLASS) NN::Sub::IListener;

%feature("director") Base;
%feature("director") Sub;
%feature("director") Base::Nest;

%inline %{
namespace NN {
class Base {
public:
        virtual ~Base(){}
        class Nest {
        public:
                virtual ~Nest(){}
                virtual bool GetValue(){ return false; }
        };
protected:
        virtual bool DoNothing() = 0;
};

class Sub : public Base {
public:
        class IListener {
        };
public:
        virtual ~Sub(){}
protected:
        void DoSomething(){}
        virtual bool GetValue() const { return true; }
};
}
%}
