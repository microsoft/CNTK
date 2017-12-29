/* Test the tp_richcompare functions generated with the -builtin option */

%module python_richcompare

%inline {

class BaseClass {
public:
    BaseClass (int i_) : i(i_) {}
    ~BaseClass () {}

    int getValue () const
    { return i; }
    
    bool operator< (const BaseClass& x) const
    { return this->i < x.i; }

    bool operator> (const BaseClass& x) const
    { return this->i > x.i; }

    bool operator<= (const BaseClass& x) const
    { return this->i <= x.i; }

    bool operator>= (const BaseClass& x) const
    { return this->i >= x.i; }

    bool operator== (const BaseClass& x) const
    { return this->i == x.i; }

    bool operator!= (const BaseClass& x) const
    { return this->i != x.i; }

    int i;
};

class SubClassA : public BaseClass {
public:
    SubClassA (int i_) : BaseClass(i_) {}
    ~SubClassA () {}

    bool operator== (const SubClassA& x) const
    { return true; }

    bool operator== (const BaseClass& x) const
    { return false; }
};

class SubClassB : public BaseClass {
public:
    SubClassB (int i_) : BaseClass(i_) {}
    ~SubClassB () {}

    bool operator== (const SubClassB& x) const
    { return true; }

    bool operator== (const SubClassA& x) const
    { return false; }
};

}
