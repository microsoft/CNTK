%module xxx

%namewarn("314:'key1' is a keyword, renaming to '_key1'", rename="_%s") "key1";
%namewarn("314:'key2' is a keyword, renaming to '_key2'", rename="_%s") "key2";
%namewarn("314:'key3' is a keyword, renaming to '_key3'", rename="_%s") "key3";
%namewarn("314:'key4' is a keyword, renaming to '_key4'", rename="_%s") "key4";
%namewarn("314:'key5' is a keyword, renaming to '_key5'", rename="_%s") "key5";

// Non-templated
%ignore KlassA::key1;
%rename(key2renamed) KlassA::key2;
%rename(key3renamed) KlassA::key3;
%rename(key4renamed) KlassA::key4;

// Templated
%ignore KlassB::key1;
%rename(key2renamed) KlassB::key2;
%rename(key3renamed) KlassB<double>::key3;

// Template specialized
%ignore KlassC::key1;
%rename(key2renamed) KlassC::key2;
%rename(key3renamed) KlassC<double>::key3;

// No warnings for these...
%inline %{
struct KlassA {
    void key1() {}
    void key2() {}
    void key3() {}
    template<typename X> void key4(X x) {}
};

template<class T> struct KlassB {
    void key1() {}
    void key2() {}
    void key3() {}
};

template<class T> struct KlassC {};
template<> struct KlassC<double> {
    void key1() {}
    void key2() {}
    void key3() {}
};

template<typename T> void key5(T t) {}

%}

%template(KlassBDouble) KlassB<double>;
%template(KlassCInt) KlassC<double>;
%template(key5renamed) key5<double>;

// These should create a single warning for each keyword...
%inline %{
struct ClassA {
    void key1() {}
    void key2() {}
    void key3() {}
    template<typename X> void key4(X x) {}
};

template<class T> struct ClassB {
    void key1() {}
    void key2() {}
    void key3() {}
};

template<class T> struct ClassC {};
template<> struct ClassC<double> {
    void key1() {}
    void key2() {}
    void key3() {}
};
%}

%template(ClassBDouble) ClassB<double>;
%template(ClassCInt) ClassC<double>;
%template(key5) key5<int>;
