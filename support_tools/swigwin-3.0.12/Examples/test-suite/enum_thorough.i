%module "enum_thorough"

// Suppress warning messages from the Ruby module for all of the
// following...

#pragma SWIG nowarn=SWIGWARN_PARSE_KEYWORD

%warnfilter(SWIGWARN_RUBY_WRONG_NAME);
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) red;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) blue;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) green;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) SpeedClass::slow;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) SpeedClass::medium;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) SpeedClass::fast;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) SpeedClass::lightning;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) TemplateClass::einstein;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) TemplateClass::galileo;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) Name::TClass::faraday;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) Name::TClass::bell;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) argh;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) eek;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) OldNameStruct::whizz;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) OldNameStruct::kerboom;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) OldNameStruct::pop;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) OldNameStruct::simple1;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) OldNameStruct::singlename1;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) OldNameStruct::doublename1;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) TreesClass::oak;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) TreesClass::fir;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) TreesClass::pine;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) curly::greasy::HairStruct::blonde;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) curly::greasy::HairStruct::ginger;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) Obscure::one;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) Obscure::onetrail;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) Obscure::two;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) Obscure::twoagain;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) Obscure::twotrail;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) Obscure::twotrailagain;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) globalinstance1;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) globalinstance2;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) globalinstance3;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) Instances::memberinstance1;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) Instances::memberinstance2;
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) Instances::memberinstance3;

%warnfilter(SWIGWARN_GO_NAME_CONFLICT);                       /* Ignoring 'one' due to Go name ('ObscureOne') conflict with 'Obscure::One' */

%inline %{

#if __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)
/* for anonymous enums */
/* dereferencing type-punned pointer will break strict-aliasing rules [-Werror=strict-aliasing] */
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

enum { AnonEnum1, AnonEnum2 = 100 };
enum { ReallyAnInteger = 200 };
//enum { AnonEnum3, AnonEnum4 } instance;
namespace AnonSpace {
  enum { AnonSpaceEnum1, AnonSpaceEnum2 = 300 };
  struct AnonStruct {
    enum { AnonStructEnum1, AnonStructEnum2 = 400 };
  };
}


enum colour { red=-1, blue, green = 10 };
colour colourTest1(colour e) { return e; }
enum colour colourTest2(enum colour e) { return e; }
const colour colourTest3(const colour e) { return e; }
const enum colour colourTest4(const enum colour e) { return e; }

colour myColour;

struct SpeedClass {
  enum speed { slow=10, medium=20, fast=30, lightning };
  typedef enum speed speedtd1;

  int                           speedTest0(int s) { return s; }
  speed                         speedTest1(speed s) { return s; }
  enum speed                    speedTest2(enum speed s) { return s; }
  const speed                   speedTest3(const speed s) { return s; }
  const enum speed              speedTest4(const enum speed s) { return s; }
  speedtd1                      speedTest5(speedtd1 s) { return s; }
  const speedtd1                speedTest6(const speedtd1 s) { return s; }
  const speed &                 speedTest7(const speed &s) { return s; }
  const enum speed &            speedTest8(const enum speed &s) { return s; }

  const colour myColour2;
  speedtd1 mySpeedtd1;
  SpeedClass() : myColour2(red), mySpeedtd1(slow) { }
private:
  SpeedClass& operator=(const SpeedClass&);
};

int                            speedTest0(int s) { return s; }
SpeedClass::speed              speedTest1(SpeedClass::speed s) { return s; }
enum SpeedClass::speed         speedTest2(enum SpeedClass::speed s) { return s; }
const SpeedClass::speed        speedTest3(const SpeedClass::speed s) { return s; }
const enum SpeedClass::speed   speedTest4(const enum SpeedClass::speed s) { return s; }
const SpeedClass::speed &      speedTest5(const SpeedClass::speed &s) { return s; }


typedef enum { NamedAnon1, NamedAnon2 } namedanon;

namedanon                       namedanonTest1(namedanon e) { return e; } 

typedef enum twonamestag { TwoNames1, TwoNames2, TwoNames3 = 33 } twonames;

twonames                        twonamesTest1(twonames e) { return e; } 
twonamestag                     twonamesTest2(twonamestag e) { return e; } 
enum twonamestag                twonamesTest3(enum twonamestag e) { return e; } 

struct TwoNamesStruct {
  typedef enum twonamestag { TwoNamesStruct1, TwoNamesStruct2 } twonames;
  twonames                      twonamesTest1(twonames e) { return e; } 
  twonamestag                   twonamesTest2(twonamestag e) { return e; } 
  enum twonamestag              twonamesTest3(enum twonamestag e) { return e; } 
};

namespace AnonSpace{
  typedef enum { NamedAnonSpace1, NamedAnonSpace2 } namedanonspace;
  namedanonspace                namedanonspaceTest1(namedanonspace e) { return e; } 
  AnonSpace::namedanonspace     namedanonspaceTest2(AnonSpace::namedanonspace e) { return e; } 
}
AnonSpace::namedanonspace       namedanonspaceTest3(AnonSpace::namedanonspace e) { return e; } 
using namespace AnonSpace;
namedanonspace                  namedanonspaceTest4(namedanonspace e) { return e; } 


template<typename T> struct TemplateClass {
  enum scientists { einstein, galileo = 10 }; 
  typedef enum scientists scientiststd1;
  typedef scientists scientiststd2;
  typedef scientiststd1 scientiststd3;
  scientists                                scientistsTest1(scientists e) { return e; }
  enum scientists                           scientistsTest2(enum scientists e) { return e; }
  const scientists                          scientistsTest3(const scientists e) { return e; }
  const enum scientists                     scientistsTest4(const enum scientists e) { return e; }
  typename TemplateClass<T>::scientists     scientistsTest5(typename TemplateClass<T>::scientists e) { return e; }
  const typename TemplateClass<T>::scientists        scientistsTest6(const typename TemplateClass<T>::scientists e) { return e; }
  enum TemplateClass<T>::scientists         scientistsTest7(enum TemplateClass<T>::scientists e) { return e; }
  const enum TemplateClass<T>::scientists   scientistsTest8(const enum TemplateClass<T>::scientists e) { return e; }
  typename TemplateClass::scientists        scientistsTest9(typename TemplateClass::scientists e) { return e; }
//  enum TemplateClass::scientists            scientistsTestA(enum TemplateClass::scientists e) { return e; }
  const typename TemplateClass::scientists  scientistsTestB(const typename TemplateClass::scientists e) { return e; }
//  const enum TemplateClass::scientists      scientistsTestC(const enum TemplateClass::scientists e) { return e; }
  scientiststd1                             scientistsTestD(scientiststd1 e) { return e; }
  scientiststd2                             scientistsTestE(scientiststd2 e) { return e; }
  scientiststd3                             scientistsTestF(scientiststd3 e) { return e; }
  typename TemplateClass<T>::scientiststd1  scientistsTestG(typename TemplateClass<T>::scientiststd1 e) { return e; }
  typename TemplateClass<T>::scientiststd2  scientistsTestH(typename TemplateClass<T>::scientiststd2 e) { return e; }
  typename TemplateClass<T>::scientiststd3  scientistsTestI(typename TemplateClass<T>::scientiststd3 e) { return e; }
  const scientists &                        scientistsTestJ(const scientists &e) { return e; }
};

TemplateClass<int>::scientists              scientistsTest1(TemplateClass<int>::scientists e) { return e; }
const TemplateClass<int>::scientists        scientistsTest2(const TemplateClass<int>::scientists e) { return e; }
enum TemplateClass<int>::scientists         scientistsTest3(enum TemplateClass<int>::scientists e) { return e; }
const enum TemplateClass<int>::scientists   scientistsTest4(const enum TemplateClass<int>::scientists e) { return e; }
TemplateClass<int>::scientiststd1           scientistsTest5(TemplateClass<int>::scientiststd1 e) { return e; }
TemplateClass<int>::scientiststd2           scientistsTest6(TemplateClass<int>::scientiststd2 e) { return e; }
TemplateClass<int>::scientiststd3           scientistsTest7(TemplateClass<int>::scientiststd3 e) { return e; }
const TemplateClass<int>::scientiststd3 &   scientistsTest8(const TemplateClass<int>::scientiststd3 &e) { return e; }


namespace Name {
template<typename T> struct TClass {
  enum scientists { faraday, bell = 20 }; 
  typedef enum scientists scientiststd1;
  typedef scientists scientiststd2;
  typedef scientiststd1 scientiststd3;
  scientists                                scientistsNameTest1(scientists e) { return e; }
  enum scientists                           scientistsNameTest2(enum scientists e) { return e; }
  const scientists                          scientistsNameTest3(const scientists e) { return e; }
  const enum scientists                     scientistsNameTest4(const enum scientists e) { return e; }
  typename TClass<T>::scientists            scientistsNameTest5(typename TClass<T>::scientists e) { return e; }
  const typename TClass<T>::scientists      scientistsNameTest6(const typename TClass<T>::scientists e) { return e; }
  enum TClass<T>::scientists                scientistsNameTest7(enum TClass<T>::scientists e) { return e; }
  const enum TClass<T>::scientists          scientistsNameTest8(const enum TClass<T>::scientists e) { return e; }
  typename TClass::scientists               scientistsNameTest9(typename TClass::scientists e) { return e; }
//  enum TClass::scientists                   scientistsNameTestA(enum TClass::scientists e) { return e; }
  const typename TClass::scientists         scientistsNameTestB(const typename TClass::scientists e) { return e; }
//  const enum TClass::scientists             scientistsNameTestC(const enum TClass::scientists e) { return e; }
  scientiststd1                             scientistsNameTestD(scientiststd1 e) { return e; }
  scientiststd2                             scientistsNameTestE(scientiststd2 e) { return e; }
  scientiststd3                             scientistsNameTestF(scientiststd3 e) { return e; }
  typename TClass<T>::scientiststd1         scientistsNameTestG(typename TClass<T>::scientiststd1 e) { return e; }
  typename TClass<T>::scientiststd2         scientistsNameTestH(typename TClass<T>::scientiststd2 e) { return e; }
  typename TClass<T>::scientiststd3         scientistsNameTestI(typename TClass<T>::scientiststd3 e) { return e; }

  typename Name::TClass<T>::scientists      scientistsNameSpaceTest1(typename Name::TClass<T>::scientists e) { return e; }
  const typename Name::TClass<T>::scientists         scientistsNameSpaceTest2(const typename Name::TClass<T>::scientists e) { return e; }
  enum Name::TClass<T>::scientists          scientistsNameSpaceTest3(enum Name::TClass<T>::scientists e) { return e; }
  const enum Name::TClass<T>::scientists    scientistsNameSpaceTest4(const enum Name::TClass<T>::scientists e) { return e; }
  typename Name::TClass<T>::scientiststd1   scientistsNameSpaceTest5(typename Name::TClass<T>::scientiststd1 e) { return e; }
  typename Name::TClass<T>::scientiststd2   scientistsNameSpaceTest6(typename Name::TClass<T>::scientiststd2 e) { return e; }
  typename Name::TClass<T>::scientiststd3   scientistsNameSpaceTest7(typename Name::TClass<T>::scientiststd3 e) { return e; }

  // Test TemplateClass::scientists rather then TClass::scientists
  ::TemplateClass<int>::scientists              scientistsOtherTest1(::TemplateClass<int>::scientists e) { return e; }
  const ::TemplateClass<int>::scientists        scientistsOtherTest2(const ::TemplateClass<int>::scientists e) { return e; }
  enum ::TemplateClass<int>::scientists         scientistsOtherTest3(enum ::TemplateClass<int>::scientists e) { return e; }
  const enum ::TemplateClass<int>::scientists   scientistsOtherTest4(const enum ::TemplateClass<int>::scientists e) { return e; }
  ::TemplateClass<int>::scientiststd1           scientistsOtherTest5(::TemplateClass<int>::scientiststd1 e) { return e; }
  ::TemplateClass<int>::scientiststd2           scientistsOtherTest6(::TemplateClass<int>::scientiststd2 e) { return e; }
  ::TemplateClass<int>::scientiststd3           scientistsOtherTest7(::TemplateClass<int>::scientiststd3 e) { return e; }
};

TClass<int>::scientists                     scientistsNameTest1(TClass<int>::scientists e) { return e; }
const TClass<int>::scientists               scientistsNameTest2(const TClass<int>::scientists e) { return e; }
enum TClass<int>::scientists                scientistsNameTest3(enum TClass<int>::scientists e) { return e; }
const enum TClass<int>::scientists          scientistsNameTest4(const enum TClass<int>::scientists e) { return e; }
TClass<int>::scientiststd1                  scientistsNameTest5(TClass<int>::scientiststd1 e) { return e; }
TClass<int>::scientiststd2                  scientistsNameTest6(TClass<int>::scientiststd2 e) { return e; }
TClass<int>::scientiststd3                  scientistsNameTest7(TClass<int>::scientiststd3 e) { return e; }

Name::TClass<int>::scientists               scientistsNameSpaceTest1(Name::TClass<int>::scientists e) { return e; }
const Name::TClass<int>::scientists         scientistsNameSpaceTest2(const Name::TClass<int>::scientists e) { return e; }
enum Name::TClass<int>::scientists          scientistsNameSpaceTest3(enum Name::TClass<int>::scientists e) { return e; }
const enum Name::TClass<int>::scientists    scientistsNameSpaceTest4(const enum Name::TClass<int>::scientists e) { return e; }
Name::TClass<int>::scientiststd1            scientistsNameSpaceTest5(Name::TClass<int>::scientiststd1 e) { return e; }
Name::TClass<int>::scientiststd2            scientistsNameSpaceTest6(Name::TClass<int>::scientiststd2 e) { return e; }
Name::TClass<int>::scientiststd3            scientistsNameSpaceTest7(Name::TClass<int>::scientiststd3 e) { return e; }
}

Name::TClass<int>::scientists               scientistsNameSpaceTest8(Name::TClass<int>::scientists e) { return e; }
const Name::TClass<int>::scientists         scientistsNameSpaceTest9(const Name::TClass<int>::scientists e) { return e; }
enum Name::TClass<int>::scientists          scientistsNameSpaceTestA(enum Name::TClass<int>::scientists e) { return e; }
const enum Name::TClass<int>::scientists    scientistsNameSpaceTestB(const enum Name::TClass<int>::scientists e) { return e; }
Name::TClass<int>::scientiststd1            scientistsNameSpaceTestC(Name::TClass<int>::scientiststd1 e) { return e; }
Name::TClass<int>::scientiststd2            scientistsNameSpaceTestD(Name::TClass<int>::scientiststd2 e) { return e; }
Name::TClass<int>::scientiststd3            scientistsNameSpaceTestE(Name::TClass<int>::scientiststd3 e) { return e; }

using namespace Name;
TClass<int>::scientists                     scientistsNameSpaceTestF(TClass<int>::scientists e) { return e; }
const TClass<int>::scientists               scientistsNameSpaceTestG(const TClass<int>::scientists e) { return e; }
enum TClass<int>::scientists                scientistsNameSpaceTestH(enum TClass<int>::scientists e) { return e; }
const enum TClass<int>::scientists          scientistsNameSpaceTestI(const enum TClass<int>::scientists e) { return e; }
TClass<int>::scientiststd1                  scientistsNameSpaceTestJ(TClass<int>::scientiststd1 e) { return e; }
TClass<int>::scientiststd2                  scientistsNameSpaceTestK(TClass<int>::scientiststd2 e) { return e; }
TClass<int>::scientiststd3                  scientistsNameSpaceTestL(TClass<int>::scientiststd3 e) { return e; }

%}

%template(TemplateClassInt) TemplateClass<int>;
%template(TClassInt) Name::TClass<int>;


// %rename tests
%rename(NewNameStruct) OldNameStruct;
%rename(newname) old_name;
%rename(doublenamerenamed) doublename;
%rename(simplerenamed) simple;
%rename(singlenamerenamed) singlename;
%rename(bang) OldNameStruct::kerboom;

%inline %{
enum old_name { argh, eek = -1 };
typedef old_name old_nametd;
old_name                            renameTest1(old_name e) { return e; }
old_nametd                          renameTest2(old_nametd e) { return e; }

struct OldNameStruct {
  enum enumeration {whizz, kerboom, pop};
  enumeration                       renameTest1(enumeration e) { return e; }
  OldNameStruct::enumeration        renameTest2(OldNameStruct::enumeration e) { return e; }

  enum simple {simple1};
  typedef enum doublenametag {doublename1} doublename;
  typedef enum {singlename1} singlename;

  simple                            renameTest3(simple e) { return e; }
  doublename                        renameTest4(doublename e) { return e; }
  doublenametag                     renameTest5(doublenametag e) { return e; }
  singlename                        renameTest6(singlename e) { return e; }
};

OldNameStruct::enumeration          renameTest3(OldNameStruct::enumeration e) { return e; }
OldNameStruct::simple               renameTest4(OldNameStruct::simple e) { return e; }
OldNameStruct::doublename           renameTest5(OldNameStruct::doublename e) { return e; }
OldNameStruct::doublenametag        renameTest6(OldNameStruct::doublenametag e) { return e; }
OldNameStruct::singlename           renameTest7(OldNameStruct::singlename e) { return e; }
%}

%rename(Clash1_di1) Clash1::di1;
%rename(Clash1_di2) Clash1::di2;
%rename(Clash1_di3) Clash1::di3;
%inline %{
namespace Clash1 {
  enum DuplicateItems1 { di1, di2 = 10, di3 };
}
namespace Clash2 {
  enum DuplicateItems2 { di1, di2 = 10, di3 };
}
%}

%inline %{
struct TreesClass {
  enum trees {oak, fir, pine };
  typedef enum trees treestd1;
  typedef trees treestd2;
  typedef treestd1 treestd3;
  typedef TreesClass::trees treestd4;
  typedef treestd1 treestd5;

  trees                             treesTest1(trees e) { return e; }
  treestd1                          treesTest2(treestd1 e) { return e; }
  treestd2                          treesTest3(treestd2 e) { return e; }
  treestd3                          treesTest4(treestd3 e) { return e; }
  treestd4                          treesTest5(treestd4 e) { return e; }
  treestd5                          treesTest6(treestd5 e) { return e; }
  const trees                       treesTest7(const trees e) { return e; }
  const treestd1                    treesTest8(const treestd1 e) { return e; }
  const treestd2                    treesTest9(const treestd2 e) { return e; }
  const treestd3                    treesTestA(const treestd3 e) { return e; }
  const treestd4                    treesTestB(const treestd4 e) { return e; }
  const treestd5                    treesTestC(const treestd5 e) { return e; }
  TreesClass::trees                 treesTestD(TreesClass::trees e) { return e; }
  TreesClass::treestd1              treesTestE(TreesClass::treestd1 e) { return e; }
  TreesClass::treestd2              treesTestF(TreesClass::treestd2 e) { return e; }
  TreesClass::treestd3              treesTestG(TreesClass::treestd3 e) { return e; }
  TreesClass::treestd4              treesTestH(TreesClass::treestd4 e) { return e; }
  TreesClass::treestd5              treesTestI(TreesClass::treestd5 e) { return e; }
  const TreesClass::trees           treesTestJ(const TreesClass::trees e) { return e; }
  const TreesClass::treestd1        treesTestK(const TreesClass::treestd1 e) { return e; }
  const TreesClass::treestd2        treesTestL(const TreesClass::treestd2 e) { return e; }
  const TreesClass::treestd3        treesTestM(const TreesClass::treestd3 e) { return e; }
  const TreesClass::treestd4        treesTestN(const TreesClass::treestd4 e) { return e; }
  const TreesClass::treestd5        treesTestO(const TreesClass::treestd5 e) { return e; }
};

TreesClass::trees                   treesTest1(TreesClass::trees e) { return e; }
TreesClass::treestd1                treesTest2(TreesClass::treestd1 e) { return e; }
TreesClass::treestd2                treesTest3(TreesClass::treestd2 e) { return e; }
TreesClass::treestd3                treesTest4(TreesClass::treestd3 e) { return e; }
TreesClass::treestd4                treesTest5(TreesClass::treestd4 e) { return e; }
TreesClass::treestd5                treesTest6(TreesClass::treestd5 e) { return e; }
const TreesClass::trees             treesTest7(const TreesClass::trees e) { return e; }
const TreesClass::treestd1          treesTest8(const TreesClass::treestd1 e) { return e; }
const TreesClass::treestd2          treesTest9(const TreesClass::treestd2 e) { return e; }
const TreesClass::treestd3          treesTestA(const TreesClass::treestd3 e) { return e; }
const TreesClass::treestd4          treesTestB(const TreesClass::treestd4 e) { return e; }
const TreesClass::treestd5          treesTestC(const TreesClass::treestd5 e) { return e; }

typedef enum TreesClass::trees treesglobaltd1;
typedef TreesClass::trees treesglobaltd2;
typedef TreesClass::treestd1 treesglobaltd3;
typedef TreesClass::treestd2 treesglobaltd4;
typedef treesglobaltd4 treesglobaltd5;

treesglobaltd1                      treesTestD(treesglobaltd1 e) { return e; }
treesglobaltd2                      treesTestE(treesglobaltd2 e) { return e; }
treesglobaltd3                      treesTestF(treesglobaltd3 e) { return e; }
treesglobaltd4                      treesTestG(treesglobaltd4 e) { return e; }
treesglobaltd5                      treesTestH(treesglobaltd5 e) { return e; }
const treesglobaltd1                treesTestI(const treesglobaltd1 e) { return e; }
const treesglobaltd2                treesTestJ(const treesglobaltd2 e) { return e; }
const treesglobaltd3                treesTestK(const treesglobaltd3 e) { return e; }
const treesglobaltd4                treesTestL(const treesglobaltd4 e) { return e; }
const treesglobaltd5                treesTestM(const treesglobaltd5 e) { return e; }

typedef const enum TreesClass::trees treesglobaltd6;
typedef const TreesClass::trees treesglobaltd7;
typedef const TreesClass::treestd1 treesglobaltd8;
typedef const TreesClass::treestd2 treesglobaltd9;
typedef const treesglobaltd4 treesglobaltdA;

//treesglobaltd6                      treesTestN(treesglobaltd6 e) { return e; } // casting using an int instead of treesglobaltd6
treesglobaltd7                      treesTestO(treesglobaltd7 e) { return e; }
treesglobaltd8                      treesTestP(treesglobaltd8 e) { return e; }
treesglobaltd9                      treesTestQ(treesglobaltd9 e) { return e; }
treesglobaltdA                      treesTestR(treesglobaltdA e) { return e; }

namespace curly {
  namespace greasy {
    struct HairStruct {
      enum hair { blonde=0xFF0, ginger };
      typedef hair hairtd1;
      typedef HairStruct::hair hairtd2;
      typedef greasy::HairStruct::hair hairtd3;
      typedef curly::greasy::HairStruct::hair hairtd4;
      typedef ::curly::greasy::HairStruct::hair hairtd5;
      typedef hairtd1 hairtd6;
      typedef HairStruct::hairtd1 hairtd7;
      typedef greasy::HairStruct::hairtd1 hairtd8;
      typedef curly::greasy::HairStruct::hairtd1 hairtd9;
      typedef ::curly::greasy::HairStruct::hairtd1 hairtdA;
      hair                          hairTest1(hair e) { return e; }
      hairtd1                       hairTest2(hairtd1 e) { return e; }
      hairtd2                       hairTest3(hairtd2 e) { return e; }
      hairtd3                       hairTest4(hairtd3 e) { return e; }
      hairtd4                       hairTest5(hairtd4 e) { return e; }
      hairtd5                       hairTest6(hairtd5 e) { return e; }
      hairtd6                       hairTest7(hairtd6 e) { return e; }
      hairtd7                       hairTest8(hairtd7 e) { return e; }
      hairtd8                       hairTest9(hairtd8 e) { return e; }
      hairtd9                       hairTestA(hairtd9 e) { return e; }
      hairtdA                       hairTestB(hairtdA e) { return e; }

      ::colour                      colourTest1(::colour e) { return e; }
      enum colour                   colourTest2(enum colour e) { return e; }
      namedanon                     namedanonTest1(namedanon e) { return e; }
      AnonSpace::namedanonspace      namedanonspaceTest1(AnonSpace::namedanonspace e) { return e; }

      treesglobaltd1                treesGlobalTest1(treesglobaltd1 e) { return e; }
      treesglobaltd2                treesGlobalTest2(treesglobaltd2 e) { return e; }
      treesglobaltd3                treesGlobalTest3(treesglobaltd3 e) { return e; }
      treesglobaltd4                treesGlobalTest4(treesglobaltd4 e) { return e; }
      treesglobaltd5                treesGlobalTest5(treesglobaltd5 e) { return e; }

    };
    HairStruct::hair                hairTest1(HairStruct::hair e) { return e; }
    HairStruct::hairtd1             hairTest2(HairStruct::hairtd1 e) { return e; }
    HairStruct::hairtd2             hairTest3(HairStruct::hairtd2 e) { return e; }
    HairStruct::hairtd3             hairTest4(HairStruct::hairtd3 e) { return e; }
    HairStruct::hairtd4             hairTest5(HairStruct::hairtd4 e) { return e; }
    HairStruct::hairtd5             hairTest6(HairStruct::hairtd5 e) { return e; }
    HairStruct::hairtd6             hairTest7(HairStruct::hairtd6 e) { return e; }
    HairStruct::hairtd7             hairTest8(HairStruct::hairtd7 e) { return e; }
    HairStruct::hairtd8             hairTest9(HairStruct::hairtd8 e) { return e; }
    HairStruct::hairtd9             hairTestA(HairStruct::hairtd9 e) { return e; }
    HairStruct::hairtdA             hairTestB(HairStruct::hairtdA e) { return e; }
    const HairStruct::hair &        hairTestC(const HairStruct::hair &e) { return e; }
  }
  greasy::HairStruct::hair          hairTestA1(greasy::HairStruct::hair e) { return e; }
  greasy::HairStruct::hairtd1       hairTestA2(greasy::HairStruct::hairtd1 e) { return e; }
  greasy::HairStruct::hairtd2       hairTestA3(greasy::HairStruct::hairtd2 e) { return e; }
  greasy::HairStruct::hairtd3       hairTestA4(greasy::HairStruct::hairtd3 e) { return e; }
  greasy::HairStruct::hairtd4       hairTestA5(greasy::HairStruct::hairtd4 e) { return e; }
  greasy::HairStruct::hairtd5       hairTestA6(greasy::HairStruct::hairtd5 e) { return e; }
  greasy::HairStruct::hairtd6       hairTestA7(greasy::HairStruct::hairtd6 e) { return e; }
  greasy::HairStruct::hairtd7       hairTestA8(greasy::HairStruct::hairtd7 e) { return e; }
  greasy::HairStruct::hairtd8       hairTestA9(greasy::HairStruct::hairtd8 e) { return e; }
  greasy::HairStruct::hairtd9       hairTestAA(greasy::HairStruct::hairtd9 e) { return e; }
  greasy::HairStruct::hairtdA       hairTestAB(greasy::HairStruct::hairtdA e) { return e; }
  const greasy::HairStruct::hairtdA &     hairTestAC(const greasy::HairStruct::hairtdA &e) { return e; }
}
curly::greasy::HairStruct::hair     hairTestB1(curly::greasy::HairStruct::hair e) { return e; }
curly::greasy::HairStruct::hairtd1  hairTestB2(curly::greasy::HairStruct::hairtd1 e) { return e; }
curly::greasy::HairStruct::hairtd2  hairTestB3(curly::greasy::HairStruct::hairtd2 e) { return e; }
curly::greasy::HairStruct::hairtd3  hairTestB4(curly::greasy::HairStruct::hairtd3 e) { return e; }
curly::greasy::HairStruct::hairtd4  hairTestB5(curly::greasy::HairStruct::hairtd4 e) { return e; }
curly::greasy::HairStruct::hairtd5  hairTestB6(curly::greasy::HairStruct::hairtd5 e) { return e; }
curly::greasy::HairStruct::hairtd6  hairTestB7(curly::greasy::HairStruct::hairtd6 e) { return e; }
curly::greasy::HairStruct::hairtd7  hairTestB8(curly::greasy::HairStruct::hairtd7 e) { return e; }
curly::greasy::HairStruct::hairtd8  hairTestB9(curly::greasy::HairStruct::hairtd8 e) { return e; }
curly::greasy::HairStruct::hairtd9  hairTestBA(curly::greasy::HairStruct::hairtd9 e) { return e; }
curly::greasy::HairStruct::hairtdA  hairTestBB(curly::greasy::HairStruct::hairtdA e) { return e; }
const curly::greasy::HairStruct::hairtdA & hairTestBC(const curly::greasy::HairStruct::hairtdA &e) { return e; }

using curly::greasy::HairStruct;
HairStruct::hair                    hairTestC1(HairStruct::hair e) { return e; }
HairStruct::hairtd1                 hairTestC2(HairStruct::hairtd1 e) { return e; }
HairStruct::hairtd2                 hairTestC3(HairStruct::hairtd2 e) { return e; }
HairStruct::hairtd3                 hairTestC4(HairStruct::hairtd3 e) { return e; }
HairStruct::hairtd4                 hairTestC5(HairStruct::hairtd4 e) { return e; }
HairStruct::hairtd5                 hairTestC6(HairStruct::hairtd5 e) { return e; }
HairStruct::hairtd6                 hairTestC7(HairStruct::hairtd6 e) { return e; }
HairStruct::hairtd7                 hairTestC8(HairStruct::hairtd7 e) { return e; }
HairStruct::hairtd8                 hairTestC9(HairStruct::hairtd8 e) { return e; }
HairStruct::hairtd9                 hairTestCA(HairStruct::hairtd9 e) { return e; }
HairStruct::hairtdA                 hairTestCB(HairStruct::hairtdA e) { return e; }
const HairStruct::hairtdA &         hairTestCC(const HairStruct::hairtdA &e) { return e; }

namespace curly {
  namespace greasy {
    struct FirStruct : HairStruct {
      hair                          hairTestFir1(hair e) { return e; }
      hairtd1                       hairTestFir2(hairtd1 e) { return e; }
      hairtd2                       hairTestFir3(hairtd2 e) { return e; }
      hairtd3                       hairTestFir4(hairtd3 e) { return e; }
      hairtd4                       hairTestFir5(hairtd4 e) { return e; }
      hairtd5                       hairTestFir6(hairtd5 e) { return e; }
      hairtd6                       hairTestFir7(hairtd6 e) { return e; }
      hairtd7                       hairTestFir8(hairtd7 e) { return e; }
      hairtd8                       hairTestFir9(hairtd8 e) { return e; }
      hairtd9                       hairTestFirA(hairtd9 e) { return e; }
      hairtdA                       hairTestFirB(hairtdA e) { return e; }
    };
  }
}

struct Obscure {
  enum Zero {};
  enum One {one};
  enum Two {two, twoagain};
#if defined(SWIG)
  // g++-4.3 warns about this
  typedef enum Empty {};
#endif
  typedef enum {} AlsoEmpty;
};

// Unnamed enum instance
extern "C" {
enum { globalinstance1, globalinstance2, globalinstance3 = 30 } GlobalInstance;
}

struct Instances {
  enum { memberinstance1, memberinstance2, memberinstance3 = 300 } MemberInstance;
  Instances() : MemberInstance(memberinstance3) {}
};

%}

// Repeated values
#if defined(SWIGJAVA)
%javaconst(1);
// needed for typesafe and proper enums only
%javaconst(0) ignoreA_three;
%javaconst(0) ignoreA_thirteen;
#elif defined(SWIGCSHARP)
// needed for typesafe enums only
#ifdef SWIG_TEST_NOCSCONST
  %csconst(0) ignoreA_three;
  %csconst(0) ignoreA_thirteen;
#endif
%csconst(1);
#endif

%ignore ignoreA_one;
%ignore ignoreA_two;
%ignore ignoreA_twelve;
%ignore ignoreA_thirty_one;

%ignore ignoreB_ten;
%ignore ignoreB_twenty;
%ignore ignoreB_thirty;
%ignore ignoreB_forty;

%ignore ignoreC_eleven;
%ignore ignoreC_thirty_one;
%ignore ignoreC_forty_one;

%ignore ignoreD_ten;
%ignore ignoreD_twenty;

%ignore ignoreE_twenty;

%inline %{
struct IgnoreTest {
  enum IgnoreA { ignoreA_zero, ignoreA_one, ignoreA_two, ignoreA_three, ignoreA_ten=10, ignoreA_eleven, ignoreA_twelve, ignoreA_thirteen, ignoreA_fourteen, ignoreA_twenty=20, ignoreA_thirty=30, ignoreA_thirty_one, ignoreA_thirty_two, ignoreA_thirty_three };
  enum IgnoreB { ignoreB_ten=10, ignoreB_eleven, ignoreB_twelve, ignoreB_twenty=20, ignoreB_thirty=30, ignoreB_thirty_one, ignoreB_thirty_two, ignoreB_forty=40, ignoreB_forty_one, ignoreB_forty_two };
  enum IgnoreC { ignoreC_ten=10, ignoreC_eleven, ignoreC_twelve, ignoreC_twenty=20, ignoreC_thirty=30, ignoreC_thirty_one, ignoreC_thirty_two, ignoreC_forty=40, ignoreC_forty_one, ignoreC_forty_two };
  enum IgnoreD { ignoreD_ten=10, ignoreD_twenty=20, ignoreD_twenty_one, ignoreD_twenty_two };
  enum IgnoreE { ignoreE_zero, ignoreE_twenty=20, ignoreE_twenty_one, ignoreE_twenty_two };
};

IgnoreTest::IgnoreA ignoreATest(IgnoreTest::IgnoreA n) { return n; }
IgnoreTest::IgnoreB ignoreBTest(IgnoreTest::IgnoreB n) { return n; }
IgnoreTest::IgnoreC ignoreCTest(IgnoreTest::IgnoreC n) { return n; }
IgnoreTest::IgnoreD ignoreDTest(IgnoreTest::IgnoreD n) { return n; }
IgnoreTest::IgnoreE ignoreETest(IgnoreTest::IgnoreE n) { return n; }
%}

%inline %{
namespace RepeatSpace {
typedef enum
{
   one = 1,
   initial = one,
   two,
   three,
   llast = three,
   end = llast
} repeat;
repeat repeatTest(repeat e) { return e; }
}
%}

%inline %{
namespace EnumWithMacro {
#define PACK(C1,C2,C3,C4) ((C1<<24)|(C2<<16)|(C3<<8)|C4)
typedef enum {
  ABCD = PACK('A','B','C','D'),
  ABCD2 = ABCD
} enumWithMacro;
enumWithMacro enumWithMacroTest(enumWithMacro e) { return e; }
}
%}

%inline %{
namespace DifferentSpace {
enum DifferentTypes {
  typeint = 10,
  typeboolfalse = false,
  typebooltrue = true,
  typebooltwo,
  typechar = 'C',
  typedefaultint,
  typecharcompound='A'+1,
  typecharcompound2='B' << 2
};
DifferentTypes differentTypesTest(DifferentTypes n) { return n; }

enum {
  global_typeint = 10,
  global_typeboolfalse = false,
  global_typebooltrue = true,
  global_typebooltwo,
  global_typechar = 'C',
  global_typedefaultint,
  global_typecharcompound='A'+1,
  global_typecharcompound2='B' << 2
};
int globalDifferentTypesTest(int n) { return n; }
}
%}

#if defined(SWIGCSHARP)
%csconstvalue("1") globalenumchar1;
%csconstvalue("'B'") globalenumcharB;
%csconstvalue("1") enumchar1;
%csconstvalue("'B'") enumcharB;
#endif
%inline %{
#if defined(__clang__)
#pragma clang diagnostic push
// Suppress: illegal character encoding in character literal
#pragma clang diagnostic ignored "-Winvalid-source-encoding"
#endif

enum {
  globalenumchar0 = '\0',
  globalenumchar1 = '\1',
  globalenumchar2 = '\n',
  globalenumcharA = 'A',
  globalenumcharB = '\102', // B
  globalenumcharC = '\x43', // C
  globalenumcharD = 0x44, // D
  globalenumcharE = 69,  // E
  globalenumcharAE1 = 'Æ', // AE (latin1 encoded)
  globalenumcharAE2 = '\306', // AE (latin1 encoded)
  globalenumcharAE3 = '\xC6' // AE (latin1 encoded)
};
enum EnumChar {
  enumchar0 = '\0',
  enumchar1 = '\1',
  enumchar2 = '\n',
  enumcharA = 'A',
  enumcharB = '\102', // B
  enumcharC = '\x43', // C
  enumcharD = 0x44, // D
  enumcharE = 69, // E
  enumcharAE1 = 'Æ', // AE (latin1 encoded)
  enumcharAE2 = '\306', // AE (latin1 encoded)
  enumcharAE3 = '\xC6' // AE (latin1 encoded)
};
struct EnumCharStruct {
  enum EnumChar {
    enumchar0 = '\0',
    enumchar1 = '\1',
    enumchar2 = '\n',
    enumcharA = 'A',
    enumcharB = '\102', // B
    enumcharC = '\x43', // C
    enumcharD = 0x44, // D
    enumcharE = 69, // E
    enumcharAE1 = 'Æ', // AE (latin1 encoded)
    enumcharAE2 = '\306', // AE (latin1 encoded)
    enumcharAE3 = '\xC6' // AE (latin1 encoded)
  };
};
%}

#if defined(SWIGJAVA)
%javaconst(0);
#elif defined(SWIGCSHARP)
%csconst(0);
#endif

%inline %{
enum {
  x_globalenumchar0 = '\0',
  x_globalenumchar1 = '\1',
  x_globalenumchar2 = '\n',
  x_globalenumcharA = 'A',
  x_globalenumcharB = '\102', // B
  x_globalenumcharC = '\x43', // C
  x_globalenumcharD = 0x44, // D
  x_globalenumcharE = 69,  // E
  x_globalenumcharAE1 = 'Æ', // AE (latin1 encoded)
  x_globalenumcharAE2 = '\306', // AE (latin1 encoded)
  x_globalenumcharAE3 = '\xC6' // AE (latin1 encoded)
};
enum X_EnumChar {
  x_enumchar0 = '\0',
  x_enumchar1 = '\1',
  x_enumchar2 = '\n',
  x_enumcharA = 'A',
  x_enumcharB = '\102', // B
  x_enumcharC = '\x43', // C
  x_enumcharD = 0x44, // D
  x_enumcharE = 69, // E
  x_enumcharAE1 = 'Æ', // AE (latin1 encoded)
  x_enumcharAE2 = '\306', // AE (latin1 encoded)
  x_enumcharAE3 = '\xC6' // AE (latin1 encoded)
};
struct X_EnumCharStruct {
  enum X_EnumChar {
    enumchar0 = '\0',
    enumchar1 = '\1',
    enumchar2 = '\n',
    enumcharA = 'A',
    enumcharB = '\102', // B
    enumcharC = '\x43', // C
    enumcharD = 0x44, // D
    enumcharE = 69, // E
    enumcharAE1 = 'Æ', // AE (latin1 encoded)
    enumcharAE2 = '\306', // AE (latin1 encoded)
    enumcharAE3 = '\xC6' // AE (latin1 encoded)
  };
};
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
%}

%inline %{
namespace DifferentSpace {
enum DifferentTypesNoConst {
  typeint_noconst = 10,
  typeboolfalse_noconst = false,
  typebooltrue_noconst = true,
  typebooltwo_noconst,
  typechar_noconst = 'C',
  typedefaultint_noconst
};

enum {
  global_typeint_noconst = 10,
  global_typeboolfalse_noconst = false,
  global_typebooltrue_noconst = true,
  global_typebooltwo_noconst,
  global_typechar_noconst = 'C',
  global_typedefaultint_noconst
};
}
%}
