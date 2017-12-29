
import enum_thorough_proper.*;

public class enum_thorough_proper_runme {

  static {
    try {
        System.loadLibrary("enum_thorough_proper");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    {
      // Anonymous enums
      int i = enum_thorough_proper.AnonEnum1;
      if (enum_thorough_proper.ReallyAnInteger != 200) throw new RuntimeException("Test Anon 1 failed");
      int j = enum_thorough_proper.AnonSpaceEnum1;
      int k = AnonStruct.AnonStructEnum1;
    }
    {
      colour red = colour.red;
      enum_thorough_proper.colourTest1(red);
      enum_thorough_proper.colourTest2(red);
      enum_thorough_proper.colourTest3(red);
      enum_thorough_proper.colourTest4(red);
      enum_thorough_proper.setMyColour(red);
    }
    {
      SpeedClass s = new SpeedClass();
      SpeedClass.speed speed = SpeedClass.speed.slow;
      if (s.speedTest1(speed) != speed) throw new RuntimeException("speedTest 1 failed");
      if (s.speedTest2(speed) != speed) throw new RuntimeException("speedTest 2 failed");
      if (s.speedTest3(speed) != speed) throw new RuntimeException("speedTest 3 failed");
      if (s.speedTest4(speed) != speed) throw new RuntimeException("speedTest 4 failed");
      if (s.speedTest5(speed) != speed) throw new RuntimeException("speedTest 5 failed");
      if (s.speedTest6(speed) != speed) throw new RuntimeException("speedTest 6 failed");
      if (s.speedTest7(speed) != speed) throw new RuntimeException("speedTest 7 failed");
      if (s.speedTest8(speed) != speed) throw new RuntimeException("speedTest 8 failed");

      if (enum_thorough_proper.speedTest1(speed) != speed) throw new RuntimeException("speedTest Global 1 failed");
      if (enum_thorough_proper.speedTest2(speed) != speed) throw new RuntimeException("speedTest Global 2 failed");
      if (enum_thorough_proper.speedTest3(speed) != speed) throw new RuntimeException("speedTest Global 3 failed");
      if (enum_thorough_proper.speedTest4(speed) != speed) throw new RuntimeException("speedTest Global 4 failed");
      if (enum_thorough_proper.speedTest5(speed) != speed) throw new RuntimeException("speedTest Global 5 failed");
    }
    {
      SpeedClass s = new SpeedClass();
      SpeedClass.speed slow = SpeedClass.speed.slow;
      SpeedClass.speed lightning = SpeedClass.speed.lightning;

      if (s.getMySpeedtd1() != slow) throw new RuntimeException("mySpeedtd1 1 failed");
      if (s.getMySpeedtd1().swigValue() != 10) throw new RuntimeException("mySpeedtd1 2 failed");

      s.setMySpeedtd1(lightning);
      if (s.getMySpeedtd1() != lightning) throw new RuntimeException("mySpeedtd1 3 failed");
      if (s.getMySpeedtd1().swigValue() != 31) throw new RuntimeException("mySpeedtd1 4 failed");
    }
    {
      if (enum_thorough_proper.namedanonTest1(namedanon.NamedAnon2) != namedanon.NamedAnon2) throw new RuntimeException("namedanonTest 1 failed");
    }
    {
      twonames val = twonames.TwoNames2;
      if (enum_thorough_proper.twonamesTest1(val) != val) throw new RuntimeException("twonamesTest 1 failed");
      if (enum_thorough_proper.twonamesTest2(val) != val) throw new RuntimeException("twonamesTest 2 failed");
      if (enum_thorough_proper.twonamesTest3(val) != val) throw new RuntimeException("twonamesTest 3 failed");
    }
    {
      TwoNamesStruct t = new TwoNamesStruct();
      TwoNamesStruct.twonames val = TwoNamesStruct.twonames.TwoNamesStruct1;
      if (t.twonamesTest1(val) != val) throw new RuntimeException("twonamesTest 1 failed");
      if (t.twonamesTest2(val) != val) throw new RuntimeException("twonamesTest 2 failed");
      if (t.twonamesTest3(val) != val) throw new RuntimeException("twonamesTest 3 failed");
    }
    {
      namedanonspace val = namedanonspace.NamedAnonSpace2;
      if (enum_thorough_proper.namedanonspaceTest1(val) != val) throw new RuntimeException("namedanonspaceTest 1 failed");
      if (enum_thorough_proper.namedanonspaceTest2(val) != val) throw new RuntimeException("namedanonspaceTest 2 failed");
      if (enum_thorough_proper.namedanonspaceTest3(val) != val) throw new RuntimeException("namedanonspaceTest 3 failed");
      if (enum_thorough_proper.namedanonspaceTest4(val) != val) throw new RuntimeException("namedanonspaceTest 4 failed");
    }
    {
      TemplateClassInt t = new TemplateClassInt();
      TemplateClassInt.scientists galileo = TemplateClassInt.scientists.galileo;

      if (t.scientistsTest1(galileo) != galileo) throw new RuntimeException("scientistsTest 1 failed");
      if (t.scientistsTest2(galileo) != galileo) throw new RuntimeException("scientistsTest 2 failed");
      if (t.scientistsTest3(galileo) != galileo) throw new RuntimeException("scientistsTest 3 failed");
      if (t.scientistsTest4(galileo) != galileo) throw new RuntimeException("scientistsTest 4 failed");
      if (t.scientistsTest5(galileo) != galileo) throw new RuntimeException("scientistsTest 5 failed");
      if (t.scientistsTest6(galileo) != galileo) throw new RuntimeException("scientistsTest 6 failed");
      if (t.scientistsTest7(galileo) != galileo) throw new RuntimeException("scientistsTest 7 failed");
      if (t.scientistsTest8(galileo) != galileo) throw new RuntimeException("scientistsTest 8 failed");
      if (t.scientistsTest9(galileo) != galileo) throw new RuntimeException("scientistsTest 9 failed");
//      if (t.scientistsTestA(galileo) != galileo) throw new RuntimeException("scientistsTest A failed");
      if (t.scientistsTestB(galileo) != galileo) throw new RuntimeException("scientistsTest B failed");
//      if (t.scientistsTestC(galileo) != galileo) throw new RuntimeException("scientistsTest C failed");
      if (t.scientistsTestD(galileo) != galileo) throw new RuntimeException("scientistsTest D failed");
      if (t.scientistsTestE(galileo) != galileo) throw new RuntimeException("scientistsTest E failed");
      if (t.scientistsTestF(galileo) != galileo) throw new RuntimeException("scientistsTest F failed");
      if (t.scientistsTestG(galileo) != galileo) throw new RuntimeException("scientistsTest G failed");
      if (t.scientistsTestH(galileo) != galileo) throw new RuntimeException("scientistsTest H failed");
      if (t.scientistsTestI(galileo) != galileo) throw new RuntimeException("scientistsTest I failed");
      if (t.scientistsTestJ(galileo) != galileo) throw new RuntimeException("scientistsTest J failed");

      if (enum_thorough_proper.scientistsTest1(galileo) != galileo) throw new RuntimeException("scientistsTest Global 1 failed");
      if (enum_thorough_proper.scientistsTest2(galileo) != galileo) throw new RuntimeException("scientistsTest Global 2 failed");
      if (enum_thorough_proper.scientistsTest3(galileo) != galileo) throw new RuntimeException("scientistsTest Global 3 failed");
      if (enum_thorough_proper.scientistsTest4(galileo) != galileo) throw new RuntimeException("scientistsTest Global 4 failed");
      if (enum_thorough_proper.scientistsTest5(galileo) != galileo) throw new RuntimeException("scientistsTest Global 5 failed");
      if (enum_thorough_proper.scientistsTest6(galileo) != galileo) throw new RuntimeException("scientistsTest Global 6 failed");
      if (enum_thorough_proper.scientistsTest7(galileo) != galileo) throw new RuntimeException("scientistsTest Global 7 failed");
      if (enum_thorough_proper.scientistsTest8(galileo) != galileo) throw new RuntimeException("scientistsTest Global 8 failed");
    }
    {
      TClassInt t = new TClassInt();
      TClassInt.scientists bell = TClassInt.scientists.bell;
      TemplateClassInt.scientists galileo = TemplateClassInt.scientists.galileo;
      if (t.scientistsNameTest1(bell) != bell) throw new RuntimeException("scientistsNameTest 1 failed");
      if (t.scientistsNameTest2(bell) != bell) throw new RuntimeException("scientistsNameTest 2 failed");
      if (t.scientistsNameTest3(bell) != bell) throw new RuntimeException("scientistsNameTest 3 failed");
      if (t.scientistsNameTest4(bell) != bell) throw new RuntimeException("scientistsNameTest 4 failed");
      if (t.scientistsNameTest5(bell) != bell) throw new RuntimeException("scientistsNameTest 5 failed");
      if (t.scientistsNameTest6(bell) != bell) throw new RuntimeException("scientistsNameTest 6 failed");
      if (t.scientistsNameTest7(bell) != bell) throw new RuntimeException("scientistsNameTest 7 failed");
      if (t.scientistsNameTest8(bell) != bell) throw new RuntimeException("scientistsNameTest 8 failed");
      if (t.scientistsNameTest9(bell) != bell) throw new RuntimeException("scientistsNameTest 9 failed");
//      if (t.scientistsNameTestA(bell) != bell) throw new RuntimeException("scientistsNameTest A failed");
      if (t.scientistsNameTestB(bell) != bell) throw new RuntimeException("scientistsNameTest B failed");
//      if (t.scientistsNameTestC(bell) != bell) throw new RuntimeException("scientistsNameTest C failed");
      if (t.scientistsNameTestD(bell) != bell) throw new RuntimeException("scientistsNameTest D failed");
      if (t.scientistsNameTestE(bell) != bell) throw new RuntimeException("scientistsNameTest E failed");
      if (t.scientistsNameTestF(bell) != bell) throw new RuntimeException("scientistsNameTest F failed");
      if (t.scientistsNameTestG(bell) != bell) throw new RuntimeException("scientistsNameTest G failed");
      if (t.scientistsNameTestH(bell) != bell) throw new RuntimeException("scientistsNameTest H failed");
      if (t.scientistsNameTestI(bell) != bell) throw new RuntimeException("scientistsNameTest I failed");

      if (t.scientistsNameSpaceTest1(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest 1 failed");
      if (t.scientistsNameSpaceTest2(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest 2 failed");
      if (t.scientistsNameSpaceTest3(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest 3 failed");
      if (t.scientistsNameSpaceTest4(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest 4 failed");
      if (t.scientistsNameSpaceTest5(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest 5 failed");
      if (t.scientistsNameSpaceTest6(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest 6 failed");
      if (t.scientistsNameSpaceTest7(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest 7 failed");

      if (t.scientistsOtherTest1(galileo) != galileo) throw new RuntimeException("scientistsOtherTest 1 failed");
      if (t.scientistsOtherTest2(galileo) != galileo) throw new RuntimeException("scientistsOtherTest 2 failed");
      if (t.scientistsOtherTest3(galileo) != galileo) throw new RuntimeException("scientistsOtherTest 3 failed");
      if (t.scientistsOtherTest4(galileo) != galileo) throw new RuntimeException("scientistsOtherTest 4 failed");
      if (t.scientistsOtherTest5(galileo) != galileo) throw new RuntimeException("scientistsOtherTest 5 failed");
      if (t.scientistsOtherTest6(galileo) != galileo) throw new RuntimeException("scientistsOtherTest 6 failed");
      if (t.scientistsOtherTest7(galileo) != galileo) throw new RuntimeException("scientistsOtherTest 7 failed");

      if (enum_thorough_proper.scientistsNameTest1(bell) != bell) throw new RuntimeException("scientistsNameTest Global 1 failed");
      if (enum_thorough_proper.scientistsNameTest2(bell) != bell) throw new RuntimeException("scientistsNameTest Global 2 failed");
      if (enum_thorough_proper.scientistsNameTest3(bell) != bell) throw new RuntimeException("scientistsNameTest Global 3 failed");
      if (enum_thorough_proper.scientistsNameTest4(bell) != bell) throw new RuntimeException("scientistsNameTest Global 4 failed");
      if (enum_thorough_proper.scientistsNameTest5(bell) != bell) throw new RuntimeException("scientistsNameTest Global 5 failed");
      if (enum_thorough_proper.scientistsNameTest6(bell) != bell) throw new RuntimeException("scientistsNameTest Global 6 failed");
      if (enum_thorough_proper.scientistsNameTest7(bell) != bell) throw new RuntimeException("scientistsNameTest Global 7 failed");

      if (enum_thorough_proper.scientistsNameSpaceTest1(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 1 failed");
      if (enum_thorough_proper.scientistsNameSpaceTest2(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 2 failed");
      if (enum_thorough_proper.scientistsNameSpaceTest3(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 3 failed");
      if (enum_thorough_proper.scientistsNameSpaceTest4(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 4 failed");
      if (enum_thorough_proper.scientistsNameSpaceTest5(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 5 failed");
      if (enum_thorough_proper.scientistsNameSpaceTest6(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 6 failed");
      if (enum_thorough_proper.scientistsNameSpaceTest7(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 7 failed");

      if (enum_thorough_proper.scientistsNameSpaceTest8(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 8 failed");
      if (enum_thorough_proper.scientistsNameSpaceTest9(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 9 failed");
      if (enum_thorough_proper.scientistsNameSpaceTestA(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global A failed");
      if (enum_thorough_proper.scientistsNameSpaceTestB(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global B failed");
      if (enum_thorough_proper.scientistsNameSpaceTestC(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global C failed");
      if (enum_thorough_proper.scientistsNameSpaceTestD(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global D failed");
      if (enum_thorough_proper.scientistsNameSpaceTestE(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global E failed");

      if (enum_thorough_proper.scientistsNameSpaceTestF(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global F failed");
      if (enum_thorough_proper.scientistsNameSpaceTestG(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global G failed");
      if (enum_thorough_proper.scientistsNameSpaceTestH(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global H failed");
      if (enum_thorough_proper.scientistsNameSpaceTestI(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global I failed");
      if (enum_thorough_proper.scientistsNameSpaceTestJ(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global J failed");
      if (enum_thorough_proper.scientistsNameSpaceTestK(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global K failed");
      if (enum_thorough_proper.scientistsNameSpaceTestL(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global L failed");
    }
    {
      newname val = newname.argh;
      if (enum_thorough_proper.renameTest1(val) != val) throw new RuntimeException("renameTest Global 1 failed");
      if (enum_thorough_proper.renameTest2(val) != val) throw new RuntimeException("renameTest Global 2 failed");
    }
    {
      NewNameStruct n = new NewNameStruct();
      if (n.renameTest1(NewNameStruct.enumeration.bang) != NewNameStruct.enumeration.bang) throw new RuntimeException("renameTest 1 failed");
      if (n.renameTest2(NewNameStruct.enumeration.bang) != NewNameStruct.enumeration.bang) throw new RuntimeException("renameTest 2 failed");
      if (n.renameTest3(NewNameStruct.simplerenamed.simple1) != NewNameStruct.simplerenamed.simple1) throw new RuntimeException("renameTest 3 failed");
      if (n.renameTest4(NewNameStruct.doublenamerenamed.doublename1) != NewNameStruct.doublenamerenamed.doublename1) throw new RuntimeException("renameTest 4 failed");
      if (n.renameTest5(NewNameStruct.doublenamerenamed.doublename1) != NewNameStruct.doublenamerenamed.doublename1) throw new RuntimeException("renameTest 5 failed");
      if (n.renameTest6(NewNameStruct.singlenamerenamed.singlename1) != NewNameStruct.singlenamerenamed.singlename1) throw new RuntimeException("renameTest 6 failed");
    }
    {
      if (enum_thorough_proper.renameTest3(NewNameStruct.enumeration.bang) != NewNameStruct.enumeration.bang) throw new RuntimeException("renameTest Global 3 failed");
      if (enum_thorough_proper.renameTest4(NewNameStruct.simplerenamed.simple1) != NewNameStruct.simplerenamed.simple1) throw new RuntimeException("renameTest Global 4 failed");
      if (enum_thorough_proper.renameTest5(NewNameStruct.doublenamerenamed.doublename1) != NewNameStruct.doublenamerenamed.doublename1) throw new RuntimeException("renameTest Global 5 failed");
      if (enum_thorough_proper.renameTest6(NewNameStruct.doublenamerenamed.doublename1) != NewNameStruct.doublenamerenamed.doublename1) throw new RuntimeException("renameTest Global 6 failed");
      if (enum_thorough_proper.renameTest7(NewNameStruct.singlenamerenamed.singlename1) != NewNameStruct.singlenamerenamed.singlename1) throw new RuntimeException("renameTest Global 7 failed");
    }
    {
      TreesClass t = new TreesClass();
      TreesClass.trees pine = TreesClass.trees.pine;

      if (t.treesTest1(pine) != pine) throw new RuntimeException("treesTest 1 failed");
      if (t.treesTest2(pine) != pine) throw new RuntimeException("treesTest 2 failed");
      if (t.treesTest3(pine) != pine) throw new RuntimeException("treesTest 3 failed");
      if (t.treesTest4(pine) != pine) throw new RuntimeException("treesTest 4 failed");
      if (t.treesTest5(pine) != pine) throw new RuntimeException("treesTest 5 failed");
      if (t.treesTest6(pine) != pine) throw new RuntimeException("treesTest 6 failed");
      if (t.treesTest7(pine) != pine) throw new RuntimeException("treesTest 7 failed");
      if (t.treesTest8(pine) != pine) throw new RuntimeException("treesTest 8 failed");
      if (t.treesTest9(pine) != pine) throw new RuntimeException("treesTest 9 failed");
      if (t.treesTestA(pine) != pine) throw new RuntimeException("treesTest A failed");
      if (t.treesTestB(pine) != pine) throw new RuntimeException("treesTest B failed");
      if (t.treesTestC(pine) != pine) throw new RuntimeException("treesTest C failed");
      if (t.treesTestD(pine) != pine) throw new RuntimeException("treesTest D failed");
      if (t.treesTestE(pine) != pine) throw new RuntimeException("treesTest E failed");
      if (t.treesTestF(pine) != pine) throw new RuntimeException("treesTest F failed");
      if (t.treesTestG(pine) != pine) throw new RuntimeException("treesTest G failed");
      if (t.treesTestH(pine) != pine) throw new RuntimeException("treesTest H failed");
      if (t.treesTestI(pine) != pine) throw new RuntimeException("treesTest I failed");
      if (t.treesTestJ(pine) != pine) throw new RuntimeException("treesTest J failed");
      if (t.treesTestK(pine) != pine) throw new RuntimeException("treesTest K failed");
      if (t.treesTestL(pine) != pine) throw new RuntimeException("treesTest L failed");
      if (t.treesTestM(pine) != pine) throw new RuntimeException("treesTest M failed");
      if (t.treesTestN(pine) != pine) throw new RuntimeException("treesTest N failed");
      if (t.treesTestO(pine) != pine) throw new RuntimeException("treesTest O failed");

      if (enum_thorough_proper.treesTest1(pine) != pine) throw new RuntimeException("treesTest Global 1 failed");
      if (enum_thorough_proper.treesTest2(pine) != pine) throw new RuntimeException("treesTest Global 2 failed");
      if (enum_thorough_proper.treesTest3(pine) != pine) throw new RuntimeException("treesTest Global 3 failed");
      if (enum_thorough_proper.treesTest4(pine) != pine) throw new RuntimeException("treesTest Global 4 failed");
      if (enum_thorough_proper.treesTest5(pine) != pine) throw new RuntimeException("treesTest Global 5 failed");
      if (enum_thorough_proper.treesTest6(pine) != pine) throw new RuntimeException("treesTest Global 6 failed");
      if (enum_thorough_proper.treesTest7(pine) != pine) throw new RuntimeException("treesTest Global 7 failed");
      if (enum_thorough_proper.treesTest8(pine) != pine) throw new RuntimeException("treesTest Global 8 failed");
      if (enum_thorough_proper.treesTest9(pine) != pine) throw new RuntimeException("treesTest Global 9 failed");
      if (enum_thorough_proper.treesTestA(pine) != pine) throw new RuntimeException("treesTest Global A failed");
      if (enum_thorough_proper.treesTestB(pine) != pine) throw new RuntimeException("treesTest Global B failed");
      if (enum_thorough_proper.treesTestC(pine) != pine) throw new RuntimeException("treesTest Global C failed");
      if (enum_thorough_proper.treesTestD(pine) != pine) throw new RuntimeException("treesTest Global D failed");
      if (enum_thorough_proper.treesTestE(pine) != pine) throw new RuntimeException("treesTest Global E failed");
      if (enum_thorough_proper.treesTestF(pine) != pine) throw new RuntimeException("treesTest Global F failed");
      if (enum_thorough_proper.treesTestG(pine) != pine) throw new RuntimeException("treesTest Global G failed");
      if (enum_thorough_proper.treesTestH(pine) != pine) throw new RuntimeException("treesTest Global H failed");
      if (enum_thorough_proper.treesTestI(pine) != pine) throw new RuntimeException("treesTest Global I failed");
      if (enum_thorough_proper.treesTestJ(pine) != pine) throw new RuntimeException("treesTest Global J failed");
      if (enum_thorough_proper.treesTestK(pine) != pine) throw new RuntimeException("treesTest Global K failed");
      if (enum_thorough_proper.treesTestL(pine) != pine) throw new RuntimeException("treesTest Global L failed");
      if (enum_thorough_proper.treesTestM(pine) != pine) throw new RuntimeException("treesTest Global M failed");
//      if (enum_thorough_proper.treesTestN(pine) != pine) throw new RuntimeException("treesTest Global N failed");
      if (enum_thorough_proper.treesTestO(pine) != pine) throw new RuntimeException("treesTest Global O failed");
      if (enum_thorough_proper.treesTestP(pine) != pine) throw new RuntimeException("treesTest Global P failed");
      if (enum_thorough_proper.treesTestQ(pine) != pine) throw new RuntimeException("treesTest Global Q failed");
      if (enum_thorough_proper.treesTestR(pine) != pine) throw new RuntimeException("treesTest Global R failed");
    }
    {
      HairStruct h = new HairStruct();
      HairStruct.hair ginger = HairStruct.hair.ginger;

      if (h.hairTest1(ginger) != ginger) throw new RuntimeException("hairTest 1 failed");
      if (h.hairTest2(ginger) != ginger) throw new RuntimeException("hairTest 2 failed");
      if (h.hairTest3(ginger) != ginger) throw new RuntimeException("hairTest 3 failed");
      if (h.hairTest4(ginger) != ginger) throw new RuntimeException("hairTest 4 failed");
      if (h.hairTest5(ginger) != ginger) throw new RuntimeException("hairTest 5 failed");
      if (h.hairTest6(ginger) != ginger) throw new RuntimeException("hairTest 6 failed");
      if (h.hairTest7(ginger) != ginger) throw new RuntimeException("hairTest 7 failed");
      if (h.hairTest8(ginger) != ginger) throw new RuntimeException("hairTest 8 failed");
      if (h.hairTest9(ginger) != ginger) throw new RuntimeException("hairTest 9 failed");
      if (h.hairTestA(ginger) != ginger) throw new RuntimeException("hairTest A failed");
      if (h.hairTestB(ginger) != ginger) throw new RuntimeException("hairTest B failed");

      colour red = colour.red;
      if (h.colourTest1(red) != red) throw new RuntimeException("colourTest HairStruct 1 failed");
      if (h.colourTest2(red) != red) throw new RuntimeException("colourTest HairStruct 2 failed");
      if (h.namedanonTest1(namedanon.NamedAnon2) != namedanon.NamedAnon2) throw new RuntimeException("namedanonTest HairStruct 1 failed");
      if (h.namedanonspaceTest1(namedanonspace.NamedAnonSpace2) != namedanonspace.NamedAnonSpace2) throw new RuntimeException("namedanonspaceTest HairStruct 1 failed");

      TreesClass.trees fir = TreesClass.trees.fir;
      if (h.treesGlobalTest1(fir) != fir) throw new RuntimeException("treesGlobalTest1 HairStruct 1 failed");
      if (h.treesGlobalTest2(fir) != fir) throw new RuntimeException("treesGlobalTest1 HairStruct 2 failed");
      if (h.treesGlobalTest3(fir) != fir) throw new RuntimeException("treesGlobalTest1 HairStruct 3 failed");
      if (h.treesGlobalTest4(fir) != fir) throw new RuntimeException("treesGlobalTest1 HairStruct 4 failed");
    }
    {
      HairStruct.hair blonde = HairStruct.hair.blonde;
      if (enum_thorough_proper.hairTest1(blonde) != blonde) throw new RuntimeException("hairTest Global 1 failed");
      if (enum_thorough_proper.hairTest2(blonde) != blonde) throw new RuntimeException("hairTest Global 2 failed");
      if (enum_thorough_proper.hairTest3(blonde) != blonde) throw new RuntimeException("hairTest Global 3 failed");
      if (enum_thorough_proper.hairTest4(blonde) != blonde) throw new RuntimeException("hairTest Global 4 failed");
      if (enum_thorough_proper.hairTest5(blonde) != blonde) throw new RuntimeException("hairTest Global 5 failed");
      if (enum_thorough_proper.hairTest6(blonde) != blonde) throw new RuntimeException("hairTest Global 6 failed");
      if (enum_thorough_proper.hairTest7(blonde) != blonde) throw new RuntimeException("hairTest Global 7 failed");
      if (enum_thorough_proper.hairTest8(blonde) != blonde) throw new RuntimeException("hairTest Global 8 failed");
      if (enum_thorough_proper.hairTest9(blonde) != blonde) throw new RuntimeException("hairTest Global 9 failed");
      if (enum_thorough_proper.hairTestA(blonde) != blonde) throw new RuntimeException("hairTest Global A failed");
      if (enum_thorough_proper.hairTestB(blonde) != blonde) throw new RuntimeException("hairTest Global B failed");
      if (enum_thorough_proper.hairTestC(blonde) != blonde) throw new RuntimeException("hairTest Global C failed");

      if (enum_thorough_proper.hairTestA1(blonde) != blonde) throw new RuntimeException("hairTest Global A1 failed");
      if (enum_thorough_proper.hairTestA2(blonde) != blonde) throw new RuntimeException("hairTest Global A2 failed");
      if (enum_thorough_proper.hairTestA3(blonde) != blonde) throw new RuntimeException("hairTest Global A3 failed");
      if (enum_thorough_proper.hairTestA4(blonde) != blonde) throw new RuntimeException("hairTest Global A4 failed");
      if (enum_thorough_proper.hairTestA5(blonde) != blonde) throw new RuntimeException("hairTest Global A5 failed");
      if (enum_thorough_proper.hairTestA6(blonde) != blonde) throw new RuntimeException("hairTest Global A6 failed");
      if (enum_thorough_proper.hairTestA7(blonde) != blonde) throw new RuntimeException("hairTest Global A7 failed");
      if (enum_thorough_proper.hairTestA8(blonde) != blonde) throw new RuntimeException("hairTest Global A8 failed");
      if (enum_thorough_proper.hairTestA9(blonde) != blonde) throw new RuntimeException("hairTest Global A9 failed");
      if (enum_thorough_proper.hairTestAA(blonde) != blonde) throw new RuntimeException("hairTest Global AA failed");
      if (enum_thorough_proper.hairTestAB(blonde) != blonde) throw new RuntimeException("hairTest Global AB failed");
      if (enum_thorough_proper.hairTestAC(blonde) != blonde) throw new RuntimeException("hairTest Global AC failed");

      if (enum_thorough_proper.hairTestB1(blonde) != blonde) throw new RuntimeException("hairTest Global B1 failed");
      if (enum_thorough_proper.hairTestB2(blonde) != blonde) throw new RuntimeException("hairTest Global B2 failed");
      if (enum_thorough_proper.hairTestB3(blonde) != blonde) throw new RuntimeException("hairTest Global B3 failed");
      if (enum_thorough_proper.hairTestB4(blonde) != blonde) throw new RuntimeException("hairTest Global B4 failed");
      if (enum_thorough_proper.hairTestB5(blonde) != blonde) throw new RuntimeException("hairTest Global B5 failed");
      if (enum_thorough_proper.hairTestB6(blonde) != blonde) throw new RuntimeException("hairTest Global B6 failed");
      if (enum_thorough_proper.hairTestB7(blonde) != blonde) throw new RuntimeException("hairTest Global B7 failed");
      if (enum_thorough_proper.hairTestB8(blonde) != blonde) throw new RuntimeException("hairTest Global B8 failed");
      if (enum_thorough_proper.hairTestB9(blonde) != blonde) throw new RuntimeException("hairTest Global B9 failed");
      if (enum_thorough_proper.hairTestBA(blonde) != blonde) throw new RuntimeException("hairTest Global BA failed");
      if (enum_thorough_proper.hairTestBB(blonde) != blonde) throw new RuntimeException("hairTest Global BB failed");
      if (enum_thorough_proper.hairTestBC(blonde) != blonde) throw new RuntimeException("hairTest Global BC failed");

      if (enum_thorough_proper.hairTestC1(blonde) != blonde) throw new RuntimeException("hairTest Global C1 failed");
      if (enum_thorough_proper.hairTestC2(blonde) != blonde) throw new RuntimeException("hairTest Global C2 failed");
      if (enum_thorough_proper.hairTestC3(blonde) != blonde) throw new RuntimeException("hairTest Global C3 failed");
      if (enum_thorough_proper.hairTestC4(blonde) != blonde) throw new RuntimeException("hairTest Global C4 failed");
      if (enum_thorough_proper.hairTestC5(blonde) != blonde) throw new RuntimeException("hairTest Global C5 failed");
      if (enum_thorough_proper.hairTestC6(blonde) != blonde) throw new RuntimeException("hairTest Global C6 failed");
      if (enum_thorough_proper.hairTestC7(blonde) != blonde) throw new RuntimeException("hairTest Global C7 failed");
      if (enum_thorough_proper.hairTestC8(blonde) != blonde) throw new RuntimeException("hairTest Global C8 failed");
      if (enum_thorough_proper.hairTestC9(blonde) != blonde) throw new RuntimeException("hairTest Global C9 failed");
      if (enum_thorough_proper.hairTestCA(blonde) != blonde) throw new RuntimeException("hairTest Global CA failed");
      if (enum_thorough_proper.hairTestCB(blonde) != blonde) throw new RuntimeException("hairTest Global CB failed");
      if (enum_thorough_proper.hairTestCC(blonde) != blonde) throw new RuntimeException("hairTest Global CC failed");
    }
    {
      FirStruct f = new FirStruct();
      HairStruct.hair blonde = HairStruct.hair.blonde;

      if (f.hairTestFir1(blonde) != blonde) throw new RuntimeException("hairTestFir 1 failed");
      if (f.hairTestFir2(blonde) != blonde) throw new RuntimeException("hairTestFir 2 failed");
      if (f.hairTestFir3(blonde) != blonde) throw new RuntimeException("hairTestFir 3 failed");
      if (f.hairTestFir4(blonde) != blonde) throw new RuntimeException("hairTestFir 4 failed");
      if (f.hairTestFir5(blonde) != blonde) throw new RuntimeException("hairTestFir 5 failed");
      if (f.hairTestFir6(blonde) != blonde) throw new RuntimeException("hairTestFir 6 failed");
      if (f.hairTestFir7(blonde) != blonde) throw new RuntimeException("hairTestFir 7 failed");
      if (f.hairTestFir8(blonde) != blonde) throw new RuntimeException("hairTestFir 8 failed");
      if (f.hairTestFir9(blonde) != blonde) throw new RuntimeException("hairTestFir 9 failed");
      if (f.hairTestFirA(blonde) != blonde) throw new RuntimeException("hairTestFir A failed");
    }
    {
      enum_thorough_proper.setGlobalInstance(enum_thorough_proper.globalinstance2);
      if (enum_thorough_proper.getGlobalInstance() != enum_thorough_proper.globalinstance2) throw new RuntimeException("GlobalInstance 1 failed");

      Instances i = new Instances();
      i.setMemberInstance(Instances.memberinstance3);
      if (i.getMemberInstance() != Instances.memberinstance3) throw new RuntimeException("MemberInstance 1 failed");
    }
    // ignore enum item tests start
    {
      if (enum_thorough_proper.ignoreATest(IgnoreTest.IgnoreA.ignoreA_zero).swigValue() != 0) throw new RuntimeException("ignoreATest 0 failed");
      if (enum_thorough_proper.ignoreATest(IgnoreTest.IgnoreA.ignoreA_three).swigValue() != 3) throw new RuntimeException("ignoreATest 3 failed");
      if (enum_thorough_proper.ignoreATest(IgnoreTest.IgnoreA.ignoreA_ten).swigValue() != 10) throw new RuntimeException("ignoreATest 10 failed");
      if (enum_thorough_proper.ignoreATest(IgnoreTest.IgnoreA.ignoreA_eleven).swigValue() != 11) throw new RuntimeException("ignoreATest 11 failed");
      if (enum_thorough_proper.ignoreATest(IgnoreTest.IgnoreA.ignoreA_thirteen).swigValue() != 13) throw new RuntimeException("ignoreATest 13 failed");
      if (enum_thorough_proper.ignoreATest(IgnoreTest.IgnoreA.ignoreA_fourteen).swigValue() != 14) throw new RuntimeException("ignoreATest 14 failed");
      if (enum_thorough_proper.ignoreATest(IgnoreTest.IgnoreA.ignoreA_twenty).swigValue() != 20) throw new RuntimeException("ignoreATest 20 failed");
      if (enum_thorough_proper.ignoreATest(IgnoreTest.IgnoreA.ignoreA_thirty).swigValue() != 30) throw new RuntimeException("ignoreATest 30 failed");
      if (enum_thorough_proper.ignoreATest(IgnoreTest.IgnoreA.ignoreA_thirty_two).swigValue() != 32) throw new RuntimeException("ignoreATest 32 failed");
      if (enum_thorough_proper.ignoreATest(IgnoreTest.IgnoreA.ignoreA_thirty_three).swigValue() != 33) throw new RuntimeException("ignoreATest 33 failed");
    }                                                         
    {                                                         
      if (enum_thorough_proper.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_eleven).swigValue() != 11) throw new RuntimeException("ignoreBTest 11 failed");
      if (enum_thorough_proper.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_twelve).swigValue() != 12) throw new RuntimeException("ignoreBTest 12 failed");
      if (enum_thorough_proper.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_thirty_one).swigValue() != 31) throw new RuntimeException("ignoreBTest 31 failed");
      if (enum_thorough_proper.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_thirty_two).swigValue() != 32) throw new RuntimeException("ignoreBTest 32 failed");
      if (enum_thorough_proper.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_forty_one).swigValue() != 41) throw new RuntimeException("ignoreBTest 41 failed");
      if (enum_thorough_proper.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_forty_two).swigValue() != 42) throw new RuntimeException("ignoreBTest 42 failed");
    }                                                         
    {                                                         
      if (enum_thorough_proper.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_ten).swigValue() != 10) throw new RuntimeException("ignoreCTest 10 failed");
      if (enum_thorough_proper.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_twelve).swigValue() != 12) throw new RuntimeException("ignoreCTest 12 failed");
      if (enum_thorough_proper.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_thirty).swigValue() != 30) throw new RuntimeException("ignoreCTest 30 failed");
      if (enum_thorough_proper.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_thirty_two).swigValue() != 32) throw new RuntimeException("ignoreCTest 32 failed");
      if (enum_thorough_proper.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_forty).swigValue() != 40) throw new RuntimeException("ignoreCTest 40 failed");
      if (enum_thorough_proper.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_forty_two).swigValue() != 42) throw new RuntimeException("ignoreCTest 42 failed");
    }                                                         
    {                                                         
      if (enum_thorough_proper.ignoreDTest(IgnoreTest.IgnoreD.ignoreD_twenty_one).swigValue() != 21) throw new RuntimeException("ignoreDTest 21 failed");
      if (enum_thorough_proper.ignoreDTest(IgnoreTest.IgnoreD.ignoreD_twenty_two).swigValue() != 22) throw new RuntimeException("ignoreDTest 22 failed");
    }                                                         
    {                                                         
      if (enum_thorough_proper.ignoreETest(IgnoreTest.IgnoreE.ignoreE_zero).swigValue() != 0) throw new RuntimeException("ignoreETest 0 failed");
      if (enum_thorough_proper.ignoreETest(IgnoreTest.IgnoreE.ignoreE_twenty_one).swigValue() != 21) throw new RuntimeException("ignoreETest 21 failed");
      if (enum_thorough_proper.ignoreETest(IgnoreTest.IgnoreE.ignoreE_twenty_two).swigValue() != 22) throw new RuntimeException("ignoreETest 22 failed");
    }
    // ignore enum item tests end
    {
      if (enum_thorough_proper.repeatTest(repeat.one).swigValue() != 1) throw new RuntimeException("repeatTest 1 failed");
      if (enum_thorough_proper.repeatTest(repeat.initial).swigValue() != 1) throw new RuntimeException("repeatTest 2 failed");
      if (enum_thorough_proper.repeatTest(repeat.two).swigValue() != 2) throw new RuntimeException("repeatTest 3 failed");
      if (enum_thorough_proper.repeatTest(repeat.three).swigValue() != 3) throw new RuntimeException("repeatTest 4 failed");
      if (enum_thorough_proper.repeatTest(repeat.llast).swigValue() != 3) throw new RuntimeException("repeatTest 5 failed");
      if (enum_thorough_proper.repeatTest(repeat.end).swigValue() != 3) throw new RuntimeException("repeatTest 6 failed");
    }
    // different types
    {
      if (enum_thorough_proper.differentTypesTest(DifferentTypes.typeint).swigValue() != 10) throw new RuntimeException("differentTypes 1 failed");
      if (enum_thorough_proper.differentTypesTest(DifferentTypes.typeboolfalse).swigValue() != 0) throw new RuntimeException("differentTypes 2 failed");
      if (enum_thorough_proper.differentTypesTest(DifferentTypes.typebooltrue).swigValue() != 1) throw new RuntimeException("differentTypes 3 failed");
      if (enum_thorough_proper.differentTypesTest(DifferentTypes.typebooltwo).swigValue() != 2) throw new RuntimeException("differentTypes 4 failed");
      if (enum_thorough_proper.differentTypesTest(DifferentTypes.typechar).swigValue() != 'C') throw new RuntimeException("differentTypes 5 failed");
      if (enum_thorough_proper.differentTypesTest(DifferentTypes.typedefaultint).swigValue() != 'D') throw new RuntimeException("differentTypes 6 failed");

      int global_enum = enum_thorough_proper.global_typeint;
      if (enum_thorough_proper.globalDifferentTypesTest(global_enum) != 10) throw new RuntimeException("global differentTypes 1 failed");
      global_enum = enum_thorough_proper.global_typeboolfalse;
      if (enum_thorough_proper.globalDifferentTypesTest(global_enum) != 0) throw new RuntimeException("global differentTypes 2 failed");
      global_enum = enum_thorough_proper.global_typebooltrue;
      if (enum_thorough_proper.globalDifferentTypesTest(global_enum) != 1) throw new RuntimeException("global differentTypes 3 failed");
      global_enum = enum_thorough_proper.global_typebooltwo;
      if (enum_thorough_proper.globalDifferentTypesTest(global_enum) != 2) throw new RuntimeException("global differentTypes 4 failed");
      global_enum = enum_thorough_proper.global_typechar;
      if (enum_thorough_proper.globalDifferentTypesTest(global_enum) != 'C') throw new RuntimeException("global differentTypes 5 failed");
      global_enum = enum_thorough_proper.global_typedefaultint;
      if (enum_thorough_proper.globalDifferentTypesTest(global_enum) != 'D') throw new RuntimeException("global differentTypes 6 failed");
    }
  }
}

