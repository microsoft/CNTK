%module chartest

%inline %{
#if defined(__clang__)
#pragma clang diagnostic push
// Suppress: illegal character encoding in character literal
#pragma clang diagnostic ignored "-Winvalid-source-encoding"
#endif
char printable_global_char = 'a';
char unprintable_global_char = 0x7F;

char GetPrintableChar() {
  return 'a';
}

char GetUnprintableChar() {
  return 0x7F;
}

static const char globchar0 = '\0';
static const char globchar1 = '\1';
static const char globchar2 = '\n';
static const char globcharA = 'A';
static const char globcharB = '\102'; // B
static const char globcharC = '\x43'; // C
static const char globcharD = 0x44; // D
static const char globcharE = 69; // E
static const char globcharAE1 = 'Æ'; // AE (latin1 encoded)
static const char globcharAE2 = '\306'; // AE (latin1 encoded)
static const char globcharAE3 = '\xC6'; // AE (latin1 encoded)

struct CharTestClass {
  static const char memberchar0 = '\0';
  static const char memberchar1 = '\1';
  static const char memberchar2 = '\n';
  static const char membercharA = 'A';
  static const char membercharB = '\102'; // B
  static const char membercharC = '\x43'; // C
  static const char membercharD = 0x44; // D
  static const char membercharE = 69; // E
  static const char membercharAE1 = 'Æ'; // AE (latin1 encoded)
  static const char membercharAE2 = '\306'; // AE (latin1 encoded)
  static const char membercharAE3 = '\xC6'; // AE (latin1 encoded)
};
%}

#if defined(SWIGJAVA)
%javaconst(1);
#elif SWIGCSHARP
%csconst(1);
#elif SWIGD
%dmanifestconst;
#endif

%inline %{
static const char x_globchar0 = '\0';
static const char x_globchar1 = '\1';
static const char x_globchar2 = '\n';
static const char x_globcharA = 'A';
static const char x_globcharB = '\102'; // B
static const char x_globcharC = '\x43'; // C
static const char x_globcharD = 0x44; // D
static const char x_globcharE = 69; // E
static const char x_globcharAE1 = 'Æ'; // AE (latin1 encoded)
static const char x_globcharAE2 = '\306'; // AE (latin1 encoded)
static const char x_globcharAE3 = '\xC6'; // AE (latin1 encoded)

struct X_CharTestClass {
  static const char memberchar0 = '\0';
  static const char memberchar1 = '\1';
  static const char memberchar2 = '\n';
  static const char membercharA = 'A';
  static const char membercharB = '\102'; // B
  static const char membercharC = '\x43'; // C
  static const char membercharD = 0x44; // D
  static const char membercharE = 69; // E
  static const char membercharAE1 = 'Æ'; // AE (latin1 encoded)
  static const char membercharAE2 = '\306'; // AE (latin1 encoded)
  static const char membercharAE3 = '\xC6'; // AE (latin1 encoded)
};
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
%}
