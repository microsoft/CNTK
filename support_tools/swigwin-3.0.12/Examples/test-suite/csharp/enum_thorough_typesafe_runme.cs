using System;
using enum_thorough_typesafeNamespace;

public class runme {
  static void Main() {
    {
      // Anonymous enums
      int i = enum_thorough_typesafe.AnonEnum1;
      if (enum_thorough_typesafe.ReallyAnInteger != 200) throw new Exception("Test Anon 1 failed");
      i += enum_thorough_typesafe.AnonSpaceEnum1;
      i += AnonStruct.AnonStructEnum1;
    }
    {
      colour red = colour.red;
      enum_thorough_typesafe.colourTest1(red);
      enum_thorough_typesafe.colourTest2(red);
      enum_thorough_typesafe.colourTest3(red);
      enum_thorough_typesafe.colourTest4(red);
      enum_thorough_typesafe.myColour = red;
    }
    {
      SpeedClass s = new SpeedClass();
      SpeedClass.speed speed = SpeedClass.speed.slow;
      if (s.speedTest1(speed) != speed) throw new Exception("speedTest 1 failed");
      if (s.speedTest2(speed) != speed) throw new Exception("speedTest 2 failed");
      if (s.speedTest3(speed) != speed) throw new Exception("speedTest 3 failed");
      if (s.speedTest4(speed) != speed) throw new Exception("speedTest 4 failed");
      if (s.speedTest5(speed) != speed) throw new Exception("speedTest 5 failed");
      if (s.speedTest6(speed) != speed) throw new Exception("speedTest 6 failed");
      if (s.speedTest7(speed) != speed) throw new Exception("speedTest 7 failed");
      if (s.speedTest8(speed) != speed) throw new Exception("speedTest 8 failed");

      if (enum_thorough_typesafe.speedTest1(speed) != speed) throw new Exception("speedTest Global 1 failed");
      if (enum_thorough_typesafe.speedTest2(speed) != speed) throw new Exception("speedTest Global 2 failed");
      if (enum_thorough_typesafe.speedTest3(speed) != speed) throw new Exception("speedTest Global 3 failed");
      if (enum_thorough_typesafe.speedTest4(speed) != speed) throw new Exception("speedTest Global 4 failed");
      if (enum_thorough_typesafe.speedTest5(speed) != speed) throw new Exception("speedTest Global 5 failed");
    }
    {
      SpeedClass s = new SpeedClass();
      SpeedClass.speed slow = SpeedClass.speed.slow;
      SpeedClass.speed lightning = SpeedClass.speed.lightning;

      if (s.mySpeedtd1 != slow) throw new Exception("mySpeedtd1 1 failed");
      if (s.mySpeedtd1.swigValue != 10) throw new Exception("mySpeedtd1 2 failed");

      s.mySpeedtd1 = lightning;
      if (s.mySpeedtd1 != lightning) throw new Exception("mySpeedtd1 3 failed");
      if (s.mySpeedtd1.swigValue != 31) throw new Exception("mySpeedtd1 4 failed");
    }
    {
      if (enum_thorough_typesafe.namedanonTest1(namedanon.NamedAnon2) != namedanon.NamedAnon2) throw new Exception("namedanonTest 1 failed");
    }
    {
      twonames val = twonames.TwoNames2;
      if (enum_thorough_typesafe.twonamesTest1(val) != val) throw new Exception("twonamesTest 1 failed");
      if (enum_thorough_typesafe.twonamesTest2(val) != val) throw new Exception("twonamesTest 2 failed");
      if (enum_thorough_typesafe.twonamesTest3(val) != val) throw new Exception("twonamesTest 3 failed");
    }
    {
      TwoNamesStruct t = new TwoNamesStruct();
      TwoNamesStruct.twonames val = TwoNamesStruct.twonames.TwoNamesStruct1;
      if (t.twonamesTest1(val) != val) throw new Exception("twonamesTest 1 failed");
      if (t.twonamesTest2(val) != val) throw new Exception("twonamesTest 2 failed");
      if (t.twonamesTest3(val) != val) throw new Exception("twonamesTest 3 failed");
    }
    {
      namedanonspace val = namedanonspace.NamedAnonSpace2;
      if (enum_thorough_typesafe.namedanonspaceTest1(val) != val) throw new Exception("namedanonspaceTest 1 failed");
      if (enum_thorough_typesafe.namedanonspaceTest2(val) != val) throw new Exception("namedanonspaceTest 2 failed");
      if (enum_thorough_typesafe.namedanonspaceTest3(val) != val) throw new Exception("namedanonspaceTest 3 failed");
      if (enum_thorough_typesafe.namedanonspaceTest4(val) != val) throw new Exception("namedanonspaceTest 4 failed");
    }
    {
      TemplateClassInt t = new TemplateClassInt();
      TemplateClassInt.scientists galileo = TemplateClassInt.scientists.galileo;

      if (t.scientistsTest1(galileo) != galileo) throw new Exception("scientistsTest 1 failed");
      if (t.scientistsTest2(galileo) != galileo) throw new Exception("scientistsTest 2 failed");
      if (t.scientistsTest3(galileo) != galileo) throw new Exception("scientistsTest 3 failed");
      if (t.scientistsTest4(galileo) != galileo) throw new Exception("scientistsTest 4 failed");
      if (t.scientistsTest5(galileo) != galileo) throw new Exception("scientistsTest 5 failed");
      if (t.scientistsTest6(galileo) != galileo) throw new Exception("scientistsTest 6 failed");
      if (t.scientistsTest7(galileo) != galileo) throw new Exception("scientistsTest 7 failed");
      if (t.scientistsTest8(galileo) != galileo) throw new Exception("scientistsTest 8 failed");
      if (t.scientistsTest9(galileo) != galileo) throw new Exception("scientistsTest 9 failed");
//      if (t.scientistsTestA(galileo) != galileo) throw new Exception("scientistsTest A failed");
      if (t.scientistsTestB(galileo) != galileo) throw new Exception("scientistsTest B failed");
//      if (t.scientistsTestC(galileo) != galileo) throw new Exception("scientistsTest C failed");
      if (t.scientistsTestD(galileo) != galileo) throw new Exception("scientistsTest D failed");
      if (t.scientistsTestE(galileo) != galileo) throw new Exception("scientistsTest E failed");
      if (t.scientistsTestF(galileo) != galileo) throw new Exception("scientistsTest F failed");
      if (t.scientistsTestG(galileo) != galileo) throw new Exception("scientistsTest G failed");
      if (t.scientistsTestH(galileo) != galileo) throw new Exception("scientistsTest H failed");
      if (t.scientistsTestI(galileo) != galileo) throw new Exception("scientistsTest I failed");
      if (t.scientistsTestJ(galileo) != galileo) throw new Exception("scientistsTest J failed");

      if (enum_thorough_typesafe.scientistsTest1(galileo) != galileo) throw new Exception("scientistsTest Global 1 failed");
      if (enum_thorough_typesafe.scientistsTest2(galileo) != galileo) throw new Exception("scientistsTest Global 2 failed");
      if (enum_thorough_typesafe.scientistsTest3(galileo) != galileo) throw new Exception("scientistsTest Global 3 failed");
      if (enum_thorough_typesafe.scientistsTest4(galileo) != galileo) throw new Exception("scientistsTest Global 4 failed");
      if (enum_thorough_typesafe.scientistsTest5(galileo) != galileo) throw new Exception("scientistsTest Global 5 failed");
      if (enum_thorough_typesafe.scientistsTest6(galileo) != galileo) throw new Exception("scientistsTest Global 6 failed");
      if (enum_thorough_typesafe.scientistsTest7(galileo) != galileo) throw new Exception("scientistsTest Global 7 failed");
      if (enum_thorough_typesafe.scientistsTest8(galileo) != galileo) throw new Exception("scientistsTest Global 8 failed");
    }
    {
      TClassInt t = new TClassInt();
      TClassInt.scientists bell = TClassInt.scientists.bell;
      TemplateClassInt.scientists galileo = TemplateClassInt.scientists.galileo;
      if (t.scientistsNameTest1(bell) != bell) throw new Exception("scientistsNameTest 1 failed");
      if (t.scientistsNameTest2(bell) != bell) throw new Exception("scientistsNameTest 2 failed");
      if (t.scientistsNameTest3(bell) != bell) throw new Exception("scientistsNameTest 3 failed");
      if (t.scientistsNameTest4(bell) != bell) throw new Exception("scientistsNameTest 4 failed");
      if (t.scientistsNameTest5(bell) != bell) throw new Exception("scientistsNameTest 5 failed");
      if (t.scientistsNameTest6(bell) != bell) throw new Exception("scientistsNameTest 6 failed");
      if (t.scientistsNameTest7(bell) != bell) throw new Exception("scientistsNameTest 7 failed");
      if (t.scientistsNameTest8(bell) != bell) throw new Exception("scientistsNameTest 8 failed");
      if (t.scientistsNameTest9(bell) != bell) throw new Exception("scientistsNameTest 9 failed");
//      if (t.scientistsNameTestA(bell) != bell) throw new Exception("scientistsNameTest A failed");
      if (t.scientistsNameTestB(bell) != bell) throw new Exception("scientistsNameTest B failed");
//      if (t.scientistsNameTestC(bell) != bell) throw new Exception("scientistsNameTest C failed");
      if (t.scientistsNameTestD(bell) != bell) throw new Exception("scientistsNameTest D failed");
      if (t.scientistsNameTestE(bell) != bell) throw new Exception("scientistsNameTest E failed");
      if (t.scientistsNameTestF(bell) != bell) throw new Exception("scientistsNameTest F failed");
      if (t.scientistsNameTestG(bell) != bell) throw new Exception("scientistsNameTest G failed");
      if (t.scientistsNameTestH(bell) != bell) throw new Exception("scientistsNameTest H failed");
      if (t.scientistsNameTestI(bell) != bell) throw new Exception("scientistsNameTest I failed");

      if (t.scientistsNameSpaceTest1(bell) != bell) throw new Exception("scientistsNameSpaceTest 1 failed");
      if (t.scientistsNameSpaceTest2(bell) != bell) throw new Exception("scientistsNameSpaceTest 2 failed");
      if (t.scientistsNameSpaceTest3(bell) != bell) throw new Exception("scientistsNameSpaceTest 3 failed");
      if (t.scientistsNameSpaceTest4(bell) != bell) throw new Exception("scientistsNameSpaceTest 4 failed");
      if (t.scientistsNameSpaceTest5(bell) != bell) throw new Exception("scientistsNameSpaceTest 5 failed");
      if (t.scientistsNameSpaceTest6(bell) != bell) throw new Exception("scientistsNameSpaceTest 6 failed");
      if (t.scientistsNameSpaceTest7(bell) != bell) throw new Exception("scientistsNameSpaceTest 7 failed");

      if (t.scientistsOtherTest1(galileo) != galileo) throw new Exception("scientistsOtherTest 1 failed");
      if (t.scientistsOtherTest2(galileo) != galileo) throw new Exception("scientistsOtherTest 2 failed");
      if (t.scientistsOtherTest3(galileo) != galileo) throw new Exception("scientistsOtherTest 3 failed");
      if (t.scientistsOtherTest4(galileo) != galileo) throw new Exception("scientistsOtherTest 4 failed");
      if (t.scientistsOtherTest5(galileo) != galileo) throw new Exception("scientistsOtherTest 5 failed");
      if (t.scientistsOtherTest6(galileo) != galileo) throw new Exception("scientistsOtherTest 6 failed");
      if (t.scientistsOtherTest7(galileo) != galileo) throw new Exception("scientistsOtherTest 7 failed");

      if (enum_thorough_typesafe.scientistsNameTest1(bell) != bell) throw new Exception("scientistsNameTest Global 1 failed");
      if (enum_thorough_typesafe.scientistsNameTest2(bell) != bell) throw new Exception("scientistsNameTest Global 2 failed");
      if (enum_thorough_typesafe.scientistsNameTest3(bell) != bell) throw new Exception("scientistsNameTest Global 3 failed");
      if (enum_thorough_typesafe.scientistsNameTest4(bell) != bell) throw new Exception("scientistsNameTest Global 4 failed");
      if (enum_thorough_typesafe.scientistsNameTest5(bell) != bell) throw new Exception("scientistsNameTest Global 5 failed");
      if (enum_thorough_typesafe.scientistsNameTest6(bell) != bell) throw new Exception("scientistsNameTest Global 6 failed");
      if (enum_thorough_typesafe.scientistsNameTest7(bell) != bell) throw new Exception("scientistsNameTest Global 7 failed");

      if (enum_thorough_typesafe.scientistsNameSpaceTest1(bell) != bell) throw new Exception("scientistsNameSpaceTest Global 1 failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTest2(bell) != bell) throw new Exception("scientistsNameSpaceTest Global 2 failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTest3(bell) != bell) throw new Exception("scientistsNameSpaceTest Global 3 failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTest4(bell) != bell) throw new Exception("scientistsNameSpaceTest Global 4 failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTest5(bell) != bell) throw new Exception("scientistsNameSpaceTest Global 5 failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTest6(bell) != bell) throw new Exception("scientistsNameSpaceTest Global 6 failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTest7(bell) != bell) throw new Exception("scientistsNameSpaceTest Global 7 failed");

      if (enum_thorough_typesafe.scientistsNameSpaceTest8(bell) != bell) throw new Exception("scientistsNameSpaceTest Global 8 failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTest9(bell) != bell) throw new Exception("scientistsNameSpaceTest Global 9 failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTestA(bell) != bell) throw new Exception("scientistsNameSpaceTest Global A failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTestB(bell) != bell) throw new Exception("scientistsNameSpaceTest Global B failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTestC(bell) != bell) throw new Exception("scientistsNameSpaceTest Global C failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTestD(bell) != bell) throw new Exception("scientistsNameSpaceTest Global D failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTestE(bell) != bell) throw new Exception("scientistsNameSpaceTest Global E failed");

      if (enum_thorough_typesafe.scientistsNameSpaceTestF(bell) != bell) throw new Exception("scientistsNameSpaceTest Global F failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTestG(bell) != bell) throw new Exception("scientistsNameSpaceTest Global G failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTestH(bell) != bell) throw new Exception("scientistsNameSpaceTest Global H failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTestI(bell) != bell) throw new Exception("scientistsNameSpaceTest Global I failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTestJ(bell) != bell) throw new Exception("scientistsNameSpaceTest Global J failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTestK(bell) != bell) throw new Exception("scientistsNameSpaceTest Global K failed");
      if (enum_thorough_typesafe.scientistsNameSpaceTestL(bell) != bell) throw new Exception("scientistsNameSpaceTest Global L failed");
    }
    {
      newname val = newname.argh;
      if (enum_thorough_typesafe.renameTest1(val) != val) throw new Exception("renameTest Global 1 failed");
      if (enum_thorough_typesafe.renameTest2(val) != val) throw new Exception("renameTest Global 2 failed");
    }
    {
      NewNameStruct n = new NewNameStruct();
      if (n.renameTest1(NewNameStruct.enumeration.bang) != NewNameStruct.enumeration.bang) throw new Exception("renameTest 1 failed");
      if (n.renameTest2(NewNameStruct.enumeration.bang) != NewNameStruct.enumeration.bang) throw new Exception("renameTest 2 failed");
      if (n.renameTest3(NewNameStruct.simplerenamed.simple1) != NewNameStruct.simplerenamed.simple1) throw new Exception("renameTest 3 failed");
      if (n.renameTest4(NewNameStruct.doublenamerenamed.doublename1) != NewNameStruct.doublenamerenamed.doublename1) throw new Exception("renameTest 4 failed");
      if (n.renameTest5(NewNameStruct.doublenamerenamed.doublename1) != NewNameStruct.doublenamerenamed.doublename1) throw new Exception("renameTest 5 failed");
      if (n.renameTest6(NewNameStruct.singlenamerenamed.singlename1) != NewNameStruct.singlenamerenamed.singlename1) throw new Exception("renameTest 6 failed");
    }
    {
      if (enum_thorough_typesafe.renameTest3(NewNameStruct.enumeration.bang) != NewNameStruct.enumeration.bang) throw new Exception("renameTest Global 3 failed");
      if (enum_thorough_typesafe.renameTest4(NewNameStruct.simplerenamed.simple1) != NewNameStruct.simplerenamed.simple1) throw new Exception("renameTest Global 4 failed");
      if (enum_thorough_typesafe.renameTest5(NewNameStruct.doublenamerenamed.doublename1) != NewNameStruct.doublenamerenamed.doublename1) throw new Exception("renameTest Global 5 failed");
      if (enum_thorough_typesafe.renameTest6(NewNameStruct.doublenamerenamed.doublename1) != NewNameStruct.doublenamerenamed.doublename1) throw new Exception("renameTest Global 6 failed");
      if (enum_thorough_typesafe.renameTest7(NewNameStruct.singlenamerenamed.singlename1) != NewNameStruct.singlenamerenamed.singlename1) throw new Exception("renameTest Global 7 failed");
    }
    {
      TreesClass t = new TreesClass();
      TreesClass.trees pine = TreesClass.trees.pine;

      if (t.treesTest1(pine) != pine) throw new Exception("treesTest 1 failed");
      if (t.treesTest2(pine) != pine) throw new Exception("treesTest 2 failed");
      if (t.treesTest3(pine) != pine) throw new Exception("treesTest 3 failed");
      if (t.treesTest4(pine) != pine) throw new Exception("treesTest 4 failed");
      if (t.treesTest5(pine) != pine) throw new Exception("treesTest 5 failed");
      if (t.treesTest6(pine) != pine) throw new Exception("treesTest 6 failed");
      if (t.treesTest7(pine) != pine) throw new Exception("treesTest 7 failed");
      if (t.treesTest8(pine) != pine) throw new Exception("treesTest 8 failed");
      if (t.treesTest9(pine) != pine) throw new Exception("treesTest 9 failed");
      if (t.treesTestA(pine) != pine) throw new Exception("treesTest A failed");
      if (t.treesTestB(pine) != pine) throw new Exception("treesTest B failed");
      if (t.treesTestC(pine) != pine) throw new Exception("treesTest C failed");
      if (t.treesTestD(pine) != pine) throw new Exception("treesTest D failed");
      if (t.treesTestE(pine) != pine) throw new Exception("treesTest E failed");
      if (t.treesTestF(pine) != pine) throw new Exception("treesTest F failed");
      if (t.treesTestG(pine) != pine) throw new Exception("treesTest G failed");
      if (t.treesTestH(pine) != pine) throw new Exception("treesTest H failed");
      if (t.treesTestI(pine) != pine) throw new Exception("treesTest I failed");
      if (t.treesTestJ(pine) != pine) throw new Exception("treesTest J failed");
      if (t.treesTestK(pine) != pine) throw new Exception("treesTest K failed");
      if (t.treesTestL(pine) != pine) throw new Exception("treesTest L failed");
      if (t.treesTestM(pine) != pine) throw new Exception("treesTest M failed");
      if (t.treesTestN(pine) != pine) throw new Exception("treesTest N failed");
      if (t.treesTestO(pine) != pine) throw new Exception("treesTest O failed");

      if (enum_thorough_typesafe.treesTest1(pine) != pine) throw new Exception("treesTest Global 1 failed");
      if (enum_thorough_typesafe.treesTest2(pine) != pine) throw new Exception("treesTest Global 2 failed");
      if (enum_thorough_typesafe.treesTest3(pine) != pine) throw new Exception("treesTest Global 3 failed");
      if (enum_thorough_typesafe.treesTest4(pine) != pine) throw new Exception("treesTest Global 4 failed");
      if (enum_thorough_typesafe.treesTest5(pine) != pine) throw new Exception("treesTest Global 5 failed");
      if (enum_thorough_typesafe.treesTest6(pine) != pine) throw new Exception("treesTest Global 6 failed");
      if (enum_thorough_typesafe.treesTest7(pine) != pine) throw new Exception("treesTest Global 7 failed");
      if (enum_thorough_typesafe.treesTest8(pine) != pine) throw new Exception("treesTest Global 8 failed");
      if (enum_thorough_typesafe.treesTest9(pine) != pine) throw new Exception("treesTest Global 9 failed");
      if (enum_thorough_typesafe.treesTestA(pine) != pine) throw new Exception("treesTest Global A failed");
      if (enum_thorough_typesafe.treesTestB(pine) != pine) throw new Exception("treesTest Global B failed");
      if (enum_thorough_typesafe.treesTestC(pine) != pine) throw new Exception("treesTest Global C failed");
      if (enum_thorough_typesafe.treesTestD(pine) != pine) throw new Exception("treesTest Global D failed");
      if (enum_thorough_typesafe.treesTestE(pine) != pine) throw new Exception("treesTest Global E failed");
      if (enum_thorough_typesafe.treesTestF(pine) != pine) throw new Exception("treesTest Global F failed");
      if (enum_thorough_typesafe.treesTestG(pine) != pine) throw new Exception("treesTest Global G failed");
      if (enum_thorough_typesafe.treesTestH(pine) != pine) throw new Exception("treesTest Global H failed");
      if (enum_thorough_typesafe.treesTestI(pine) != pine) throw new Exception("treesTest Global I failed");
      if (enum_thorough_typesafe.treesTestJ(pine) != pine) throw new Exception("treesTest Global J failed");
      if (enum_thorough_typesafe.treesTestK(pine) != pine) throw new Exception("treesTest Global K failed");
      if (enum_thorough_typesafe.treesTestL(pine) != pine) throw new Exception("treesTest Global L failed");
      if (enum_thorough_typesafe.treesTestM(pine) != pine) throw new Exception("treesTest Global M failed");
//      if (enum_thorough_typesafe.treesTestN(pine) != pine) throw new Exception("treesTest Global N failed");
      if (enum_thorough_typesafe.treesTestO(pine) != pine) throw new Exception("treesTest Global O failed");
      if (enum_thorough_typesafe.treesTestP(pine) != pine) throw new Exception("treesTest Global P failed");
      if (enum_thorough_typesafe.treesTestQ(pine) != pine) throw new Exception("treesTest Global Q failed");
      if (enum_thorough_typesafe.treesTestR(pine) != pine) throw new Exception("treesTest Global R failed");
    }
    {
      HairStruct h = new HairStruct();
      HairStruct.hair ginger = HairStruct.hair.ginger;

      if (h.hairTest1(ginger) != ginger) throw new Exception("hairTest 1 failed");
      if (h.hairTest2(ginger) != ginger) throw new Exception("hairTest 2 failed");
      if (h.hairTest3(ginger) != ginger) throw new Exception("hairTest 3 failed");
      if (h.hairTest4(ginger) != ginger) throw new Exception("hairTest 4 failed");
      if (h.hairTest5(ginger) != ginger) throw new Exception("hairTest 5 failed");
      if (h.hairTest6(ginger) != ginger) throw new Exception("hairTest 6 failed");
      if (h.hairTest7(ginger) != ginger) throw new Exception("hairTest 7 failed");
      if (h.hairTest8(ginger) != ginger) throw new Exception("hairTest 8 failed");
      if (h.hairTest9(ginger) != ginger) throw new Exception("hairTest 9 failed");
      if (h.hairTestA(ginger) != ginger) throw new Exception("hairTest A failed");
      if (h.hairTestB(ginger) != ginger) throw new Exception("hairTest B failed");

      colour red = colour.red;
      if (h.colourTest1(red) != red) throw new Exception("colourTest HairStruct 1 failed");
      if (h.colourTest2(red) != red) throw new Exception("colourTest HairStruct 2 failed");
      if (h.namedanonTest1(namedanon.NamedAnon2) != namedanon.NamedAnon2) throw new Exception("namedanonTest HairStruct 1 failed");
      if (h.namedanonspaceTest1(namedanonspace.NamedAnonSpace2) != namedanonspace.NamedAnonSpace2) throw new Exception("namedanonspaceTest HairStruct 1 failed");

      TreesClass.trees fir = TreesClass.trees.fir;
      if (h.treesGlobalTest1(fir) != fir) throw new Exception("treesGlobalTest1 HairStruct 1 failed");
      if (h.treesGlobalTest2(fir) != fir) throw new Exception("treesGlobalTest1 HairStruct 2 failed");
      if (h.treesGlobalTest3(fir) != fir) throw new Exception("treesGlobalTest1 HairStruct 3 failed");
      if (h.treesGlobalTest4(fir) != fir) throw new Exception("treesGlobalTest1 HairStruct 4 failed");
    }
    {
      HairStruct.hair blonde = HairStruct.hair.blonde;
      if (enum_thorough_typesafe.hairTest1(blonde) != blonde) throw new Exception("hairTest Global 1 failed");
      if (enum_thorough_typesafe.hairTest2(blonde) != blonde) throw new Exception("hairTest Global 2 failed");
      if (enum_thorough_typesafe.hairTest3(blonde) != blonde) throw new Exception("hairTest Global 3 failed");
      if (enum_thorough_typesafe.hairTest4(blonde) != blonde) throw new Exception("hairTest Global 4 failed");
      if (enum_thorough_typesafe.hairTest5(blonde) != blonde) throw new Exception("hairTest Global 5 failed");
      if (enum_thorough_typesafe.hairTest6(blonde) != blonde) throw new Exception("hairTest Global 6 failed");
      if (enum_thorough_typesafe.hairTest7(blonde) != blonde) throw new Exception("hairTest Global 7 failed");
      if (enum_thorough_typesafe.hairTest8(blonde) != blonde) throw new Exception("hairTest Global 8 failed");
      if (enum_thorough_typesafe.hairTest9(blonde) != blonde) throw new Exception("hairTest Global 9 failed");
      if (enum_thorough_typesafe.hairTestA(blonde) != blonde) throw new Exception("hairTest Global A failed");
      if (enum_thorough_typesafe.hairTestB(blonde) != blonde) throw new Exception("hairTest Global B failed");
      if (enum_thorough_typesafe.hairTestC(blonde) != blonde) throw new Exception("hairTest Global C failed");

      if (enum_thorough_typesafe.hairTestA1(blonde) != blonde) throw new Exception("hairTest Global A1 failed");
      if (enum_thorough_typesafe.hairTestA2(blonde) != blonde) throw new Exception("hairTest Global A2 failed");
      if (enum_thorough_typesafe.hairTestA3(blonde) != blonde) throw new Exception("hairTest Global A3 failed");
      if (enum_thorough_typesafe.hairTestA4(blonde) != blonde) throw new Exception("hairTest Global A4 failed");
      if (enum_thorough_typesafe.hairTestA5(blonde) != blonde) throw new Exception("hairTest Global A5 failed");
      if (enum_thorough_typesafe.hairTestA6(blonde) != blonde) throw new Exception("hairTest Global A6 failed");
      if (enum_thorough_typesafe.hairTestA7(blonde) != blonde) throw new Exception("hairTest Global A7 failed");
      if (enum_thorough_typesafe.hairTestA8(blonde) != blonde) throw new Exception("hairTest Global A8 failed");
      if (enum_thorough_typesafe.hairTestA9(blonde) != blonde) throw new Exception("hairTest Global A9 failed");
      if (enum_thorough_typesafe.hairTestAA(blonde) != blonde) throw new Exception("hairTest Global AA failed");
      if (enum_thorough_typesafe.hairTestAB(blonde) != blonde) throw new Exception("hairTest Global AB failed");
      if (enum_thorough_typesafe.hairTestAC(blonde) != blonde) throw new Exception("hairTest Global AC failed");

      if (enum_thorough_typesafe.hairTestB1(blonde) != blonde) throw new Exception("hairTest Global B1 failed");
      if (enum_thorough_typesafe.hairTestB2(blonde) != blonde) throw new Exception("hairTest Global B2 failed");
      if (enum_thorough_typesafe.hairTestB3(blonde) != blonde) throw new Exception("hairTest Global B3 failed");
      if (enum_thorough_typesafe.hairTestB4(blonde) != blonde) throw new Exception("hairTest Global B4 failed");
      if (enum_thorough_typesafe.hairTestB5(blonde) != blonde) throw new Exception("hairTest Global B5 failed");
      if (enum_thorough_typesafe.hairTestB6(blonde) != blonde) throw new Exception("hairTest Global B6 failed");
      if (enum_thorough_typesafe.hairTestB7(blonde) != blonde) throw new Exception("hairTest Global B7 failed");
      if (enum_thorough_typesafe.hairTestB8(blonde) != blonde) throw new Exception("hairTest Global B8 failed");
      if (enum_thorough_typesafe.hairTestB9(blonde) != blonde) throw new Exception("hairTest Global B9 failed");
      if (enum_thorough_typesafe.hairTestBA(blonde) != blonde) throw new Exception("hairTest Global BA failed");
      if (enum_thorough_typesafe.hairTestBB(blonde) != blonde) throw new Exception("hairTest Global BB failed");
      if (enum_thorough_typesafe.hairTestBC(blonde) != blonde) throw new Exception("hairTest Global BC failed");

      if (enum_thorough_typesafe.hairTestC1(blonde) != blonde) throw new Exception("hairTest Global C1 failed");
      if (enum_thorough_typesafe.hairTestC2(blonde) != blonde) throw new Exception("hairTest Global C2 failed");
      if (enum_thorough_typesafe.hairTestC3(blonde) != blonde) throw new Exception("hairTest Global C3 failed");
      if (enum_thorough_typesafe.hairTestC4(blonde) != blonde) throw new Exception("hairTest Global C4 failed");
      if (enum_thorough_typesafe.hairTestC5(blonde) != blonde) throw new Exception("hairTest Global C5 failed");
      if (enum_thorough_typesafe.hairTestC6(blonde) != blonde) throw new Exception("hairTest Global C6 failed");
      if (enum_thorough_typesafe.hairTestC7(blonde) != blonde) throw new Exception("hairTest Global C7 failed");
      if (enum_thorough_typesafe.hairTestC8(blonde) != blonde) throw new Exception("hairTest Global C8 failed");
      if (enum_thorough_typesafe.hairTestC9(blonde) != blonde) throw new Exception("hairTest Global C9 failed");
      if (enum_thorough_typesafe.hairTestCA(blonde) != blonde) throw new Exception("hairTest Global CA failed");
      if (enum_thorough_typesafe.hairTestCB(blonde) != blonde) throw new Exception("hairTest Global CB failed");
      if (enum_thorough_typesafe.hairTestCC(blonde) != blonde) throw new Exception("hairTest Global CC failed");
    }
    {
      FirStruct f = new FirStruct();
      HairStruct.hair blonde = HairStruct.hair.blonde;

      if (f.hairTestFir1(blonde) != blonde) throw new Exception("hairTestFir 1 failed");
      if (f.hairTestFir2(blonde) != blonde) throw new Exception("hairTestFir 2 failed");
      if (f.hairTestFir3(blonde) != blonde) throw new Exception("hairTestFir 3 failed");
      if (f.hairTestFir4(blonde) != blonde) throw new Exception("hairTestFir 4 failed");
      if (f.hairTestFir5(blonde) != blonde) throw new Exception("hairTestFir 5 failed");
      if (f.hairTestFir6(blonde) != blonde) throw new Exception("hairTestFir 6 failed");
      if (f.hairTestFir7(blonde) != blonde) throw new Exception("hairTestFir 7 failed");
      if (f.hairTestFir8(blonde) != blonde) throw new Exception("hairTestFir 8 failed");
      if (f.hairTestFir9(blonde) != blonde) throw new Exception("hairTestFir 9 failed");
      if (f.hairTestFirA(blonde) != blonde) throw new Exception("hairTestFir A failed");
    }
    {
      enum_thorough_typesafe.GlobalInstance = enum_thorough_typesafe.globalinstance2;
      if (enum_thorough_typesafe.GlobalInstance != enum_thorough_typesafe.globalinstance2) throw new Exception("GlobalInstance 1 failed");

      Instances i = new Instances();
      i.MemberInstance = Instances.memberinstance3;
      if (i.MemberInstance != Instances.memberinstance3) throw new Exception("MemberInstance 1 failed");
    }
    // ignore enum item tests start
    {
      if (enum_thorough_typesafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_zero).swigValue != 0) throw new Exception("ignoreATest 0 failed");
      if (enum_thorough_typesafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_three).swigValue != 3) throw new Exception("ignoreATest 3 failed");
      if (enum_thorough_typesafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_ten).swigValue != 10) throw new Exception("ignoreATest 10 failed");
      if (enum_thorough_typesafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_eleven).swigValue != 11) throw new Exception("ignoreATest 11 failed");
      if (enum_thorough_typesafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_thirteen).swigValue != 13) throw new Exception("ignoreATest 13 failed");
      if (enum_thorough_typesafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_fourteen).swigValue != 14) throw new Exception("ignoreATest 14 failed");
      if (enum_thorough_typesafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_twenty).swigValue != 20) throw new Exception("ignoreATest 20 failed");
      if (enum_thorough_typesafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_thirty).swigValue != 30) throw new Exception("ignoreATest 30 failed");
      if (enum_thorough_typesafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_thirty_two).swigValue != 32) throw new Exception("ignoreATest 32 failed");
      if (enum_thorough_typesafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_thirty_three).swigValue != 33) throw new Exception("ignoreATest 33 failed");
    }                                                         
    {                                                         
      if (enum_thorough_typesafe.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_eleven).swigValue != 11) throw new Exception("ignoreBTest 11 failed");
      if (enum_thorough_typesafe.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_twelve).swigValue != 12) throw new Exception("ignoreBTest 12 failed");
      if (enum_thorough_typesafe.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_thirty_one).swigValue != 31) throw new Exception("ignoreBTest 31 failed");
      if (enum_thorough_typesafe.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_thirty_two).swigValue != 32) throw new Exception("ignoreBTest 32 failed");
      if (enum_thorough_typesafe.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_forty_one).swigValue != 41) throw new Exception("ignoreBTest 41 failed");
      if (enum_thorough_typesafe.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_forty_two).swigValue != 42) throw new Exception("ignoreBTest 42 failed");
    }                                                         
    {                                                         
      if (enum_thorough_typesafe.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_ten).swigValue != 10) throw new Exception("ignoreCTest 10 failed");
      if (enum_thorough_typesafe.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_twelve).swigValue != 12) throw new Exception("ignoreCTest 12 failed");
      if (enum_thorough_typesafe.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_thirty).swigValue != 30) throw new Exception("ignoreCTest 30 failed");
      if (enum_thorough_typesafe.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_thirty_two).swigValue != 32) throw new Exception("ignoreCTest 32 failed");
      if (enum_thorough_typesafe.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_forty).swigValue != 40) throw new Exception("ignoreCTest 40 failed");
      if (enum_thorough_typesafe.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_forty_two).swigValue != 42) throw new Exception("ignoreCTest 42 failed");
    }                                                         
    {                                                         
      if (enum_thorough_typesafe.ignoreDTest(IgnoreTest.IgnoreD.ignoreD_twenty_one).swigValue != 21) throw new Exception("ignoreDTest 21 failed");
      if (enum_thorough_typesafe.ignoreDTest(IgnoreTest.IgnoreD.ignoreD_twenty_two).swigValue != 22) throw new Exception("ignoreDTest 22 failed");
    }                                                         
    {                                                         
      if (enum_thorough_typesafe.ignoreETest(IgnoreTest.IgnoreE.ignoreE_zero).swigValue != 0) throw new Exception("ignoreETest 0 failed");
      if (enum_thorough_typesafe.ignoreETest(IgnoreTest.IgnoreE.ignoreE_twenty_one).swigValue != 21) throw new Exception("ignoreETest 21 failed");
      if (enum_thorough_typesafe.ignoreETest(IgnoreTest.IgnoreE.ignoreE_twenty_two).swigValue != 22) throw new Exception("ignoreETest 22 failed");
    }
    // ignore enum item tests end
    {
      if (enum_thorough_typesafe.repeatTest(repeat.one).swigValue != 1) throw new Exception("repeatTest 1 failed");
      if (enum_thorough_typesafe.repeatTest(repeat.initial).swigValue != 1) throw new Exception("repeatTest 2 failed");
      if (enum_thorough_typesafe.repeatTest(repeat.two).swigValue != 2) throw new Exception("repeatTest 3 failed");
      if (enum_thorough_typesafe.repeatTest(repeat.three).swigValue != 3) throw new Exception("repeatTest 4 failed");
      if (enum_thorough_typesafe.repeatTest(repeat.llast).swigValue != 3) throw new Exception("repeatTest 5 failed");
      if (enum_thorough_typesafe.repeatTest(repeat.end).swigValue != 3) throw new Exception("repeatTest 6 failed");
    }
    {
      if (enum_thorough_typesafe.enumWithMacroTest(enumWithMacro.ABCD).swigValue != (('A' << 24) | ('B' << 16) | ('C' << 8) | 'D')) throw new Exception("enumWithMacroTest 1 failed");
      if (enum_thorough_typesafe.enumWithMacroTest(enumWithMacro.ABCD2).swigValue != enum_thorough_typesafe.enumWithMacroTest(enumWithMacro.ABCD).swigValue) throw new Exception("enumWithMacroTest 2 failed");
    }
    // different types
    {
      if (enum_thorough_typesafe.differentTypesTest(DifferentTypes.typeint).swigValue != 10) throw new Exception("differentTypes 1 failed");
      if (enum_thorough_typesafe.differentTypesTest(DifferentTypes.typebooltrue).swigValue != 1) throw new Exception("differentTypes 2 failed");
      if (enum_thorough_typesafe.differentTypesTest(DifferentTypes.typebooltwo).swigValue != 2) throw new Exception("differentTypes 3 failed");
      if (enum_thorough_typesafe.differentTypesTest(DifferentTypes.typeboolfalse).swigValue != 0) throw new Exception("differentTypes 4 failed");
      if (enum_thorough_typesafe.differentTypesTest(DifferentTypes.typechar).swigValue != (int)'C') throw new Exception("differentTypes 5 failed");
      if (enum_thorough_typesafe.differentTypesTest(DifferentTypes.typedefaultint).swigValue != (int)'D') throw new Exception("differentTypes 6 failed");
      if (enum_thorough_typesafe.differentTypesTest(DifferentTypes.typecharcompound).swigValue != (int)'A' + 1) throw new Exception("differentTypes 7 failed");
      if (enum_thorough_typesafe.differentTypesTest(DifferentTypes.typecharcompound2).swigValue != (int)'B' << 2) throw new Exception("differentTypes 8 failed");

      int global_enum = enum_thorough_typesafe.global_typeint;
      if (enum_thorough_typesafe.globalDifferentTypesTest(global_enum) != 10) throw new Exception("global differentTypes 1 failed");
      global_enum = enum_thorough_typesafe.global_typeboolfalse;
      if (enum_thorough_typesafe.globalDifferentTypesTest(global_enum) != 0) throw new Exception("global differentTypes 2 failed");
      global_enum = enum_thorough_typesafe.global_typebooltrue;
      if (enum_thorough_typesafe.globalDifferentTypesTest(global_enum) != 1) throw new Exception("global differentTypes 3 failed");
      global_enum = enum_thorough_typesafe.global_typebooltwo;
      if (enum_thorough_typesafe.globalDifferentTypesTest(global_enum) != 2) throw new Exception("global differentTypes 4 failed");
      global_enum = enum_thorough_typesafe.global_typechar;
      if (enum_thorough_typesafe.globalDifferentTypesTest(global_enum) != 'C') throw new Exception("global differentTypes 5 failed");
      global_enum = enum_thorough_typesafe.global_typedefaultint;
      if (enum_thorough_typesafe.globalDifferentTypesTest(global_enum) != 'D') throw new Exception("global differentTypes 6 failed");
      global_enum = enum_thorough_typesafe.global_typecharcompound;
      if (enum_thorough_typesafe.globalDifferentTypesTest(global_enum) != (int)'A' + 1) throw new Exception("global differentTypes 7 failed");
      global_enum = enum_thorough_typesafe.global_typecharcompound2;
      if (enum_thorough_typesafe.globalDifferentTypesTest(global_enum) != (int)'B' << 2) throw new Exception("global differentTypes 8 failed");
    }
  }
}

