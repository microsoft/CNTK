
// This testcase uses the %javaconst directive to control how constants are initialised

%module java_constants


%constant short DIPSTICK=100;

// Set default Java const code generation
%javaconst(1);

// Modify the code generation to use JNI function call initialisation for some difficult cases
%javaconst(0) TOM;
%javaconst(0) ORCHESTRA_STALLS;
%javaconst(0) PORKY;

%inline %{
#define CHINA 2*100
#define TOM 300ULL
#define ORCHESTRA_STALLS 400LL
#define JAM_JAR "500"
#define OXO '6'
#define PORKY !7
%}

%constant int BRISTOLS=800;

%javaconstvalue(100L) APPLES;
%inline %{
#define APPLES 100LL
%}

%javaconst(0);
%constant long long ROSY=900LL;

