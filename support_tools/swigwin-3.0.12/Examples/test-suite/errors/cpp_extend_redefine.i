%module xxx

%extend foo {
    int bar() {
    }
};

struct foo {
    int bar();
    int spam();
};

%extend foo {
    int spam();
};


   





