/* This interface file tests whether character constants are correctly
   wrapped as procedures returning Scheme characters (rather than
   Scheme strings). 
*/

%module char_constant

#define CHAR_CONSTANT 'x'

#define STRING_CONSTANT "xyzzy"

#define ESC_CONST  '\1'
#define NULL_CONST '\0'
#define SPECIALCHAR 'á'
#define SPECIALCHAR2 '\n'
#define SPECIALCHARA 'A'
#define SPECIALCHARB '\102' // B
#define SPECIALCHARC '\x43' // C
#define SPECIALCHARD 0x44 // D
#define SPECIALCHARE 69 // E
#define SPECIALCHARAE1 'Æ' // AE (latin1 encoded)
#define SPECIALCHARAE2 '\306' // AE (latin1 encoded)
#define SPECIALCHARAE3 '\xC6' // AE (latin1 encoded)

#if defined(SWIGJAVA)
%javaconst(1);
#elif SWIGCSHARP
%csconst(1);
#elif SWIGD
%dmanifestconst;
#endif

#define X_ESC_CONST  '\1'
#define X_NULL_CONST '\0'
#define X_SPECIALCHAR 'á'
#define X_SPECIALCHAR2 '\n'
#define X_SPECIALCHARA 'A'
#define X_SPECIALCHARB '\102' // B
#define X_SPECIALCHARC '\x43' // C
#define X_SPECIALCHARD 0x44 // D
#define X_SPECIALCHARE 69 // E
#define X_SPECIALCHARAE1 'Æ' // AE (latin1 encoded)
#define X_SPECIALCHARAE2 '\306' // AE (latin1 encoded)
#define X_SPECIALCHARAE3 '\xC6' // AE (latin1 encoded)

%inline 
{
  const int ia = (int)'a';
  const int ib = 'b';
}
