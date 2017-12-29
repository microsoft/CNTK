%module sym
// make sure different classes are allowed to have methods of the same name 
// that we properly qualify wrappers in the C namespace to avoid collisions

%rename(hulahoops) Flim::Jam();

%inline %{

class Flim {
public:
   Flim() { }
   const char * Jam() { return "flim-jam"; }
   const char * Jar() { return "flim-jar"; }
};

class Flam {
public:
   Flam() { }
   const char * Jam() { return "flam-jam"; }
   const char * Jar() { return "flam-jar"; }
};

%}

