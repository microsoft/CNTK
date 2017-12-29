// A complicated test of overloaded functions
%module overload_complicated

#ifndef SWIG_NO_OVERLOAD

// Different warning filters needed for scripting languages (eg Python) and for statically typed languages (eg C#).
%warnfilter(509, 516) Pop::Pop;      // Overloaded xxxx is shadowed by xxxx at xxx:y. | Overloaded method xxxx ignored. Method at xxx:y used.
%warnfilter(509, 516) Pop::hip;      // Overloaded xxxx is shadowed by xxxx at xxx:y. | Overloaded method xxxx ignored. Method at xxx:y used.
%warnfilter(509, 516) Pop::hop;      // Overloaded xxxx is shadowed by xxxx at xxx:y. | Overloaded method xxxx ignored. Method at xxx:y used.
%warnfilter(509, 516) Pop::pop;      // Overloaded xxxx is shadowed by xxxx at xxx:y. | Overloaded method xxxx ignored. Method at xxx:y used.
%warnfilter(516) Pop::bop;           // Overloaded method xxxx ignored. Method at xxx:y used.
%warnfilter(516) Pop::bip;           // Overloaded method xxxx ignored. Method at xxx:y used.
%warnfilter(509, 516) ::muzak;       // Overloaded xxxx is shadowed by xxxx at xxx:y. | Overloaded method xxxx ignored. Method at xxx:y used.

%typemap(in, numinputs=0) int l { $1 = 4711; }

%inline %{

double foo(int, int, char *, int) {
    return 15;
}

double foo(int i, int j, double k = 17.4, int l = 18, char m = 'P') {
    return i + j + k + l + (int) m;
}

struct Pop {
    Pop(int* i) {}
    Pop(int& i) {}
    Pop(const int* i, bool b) {}
    Pop(const int* i) {}

    // Check overloaded in const only and pointers/references which target languages cannot disambiguate
    int hip(bool b)                 { return 701; }
    int hip(int* i)                 { return 702; }
    int hip(int& i)                 { return 703; }
    int hip(int* i) const           { return 704; }
    int hip(const int* i)           { return 705; }

    // Reverse the order for the above
    int hop(const int* i)           { return 805; }
    int hop(int* i) const           { return 804; }
    int hop(int& i)                 { return 803; }
    int hop(int* i)                 { return 802; }
    int hop(bool b)                 { return 801; }

    // Few more variations and order shuffled
    int pop(bool b)                 { return 901; }
    int pop(int* i) const           { return 902; }
    int pop(int& i)                 { return 903; }
    int pop(int* i)                 { return 904; }
    int pop()                       { return 905; }
    int pop(const int* i)           { return 906; }

    // Overload on const only
    int bop(int* i)                 { return 1001; }
    int bop(int* i) const           { return 1002; }

    int bip(int* i) const           { return 2001; }
    int bip(int* i)                 { return 2002; }
};

// Globals
int muzak(bool b)                 { return 3001; }
int muzak(int* i)                 { return 3002; }
int muzak(int& i)                 { return 3003; }
int muzak(const int* i)           { return 3004; }

%}

#endif

