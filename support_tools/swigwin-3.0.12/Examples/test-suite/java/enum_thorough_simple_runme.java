
import enum_thorough_simple.*;

public class enum_thorough_simple_runme {

  static {
    try {
        System.loadLibrary("enum_thorough_simple");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    {
      // Anonymous enums
      int i = enum_thorough_simple.AnonEnum1;
      if (enum_thorough_simple.ReallyAnInteger != 200) throw new RuntimeException("Test Anon 1 failed");
      int j = enum_thorough_simple.AnonSpaceEnum1;
      int k = AnonStruct.AnonStructEnum1;
    }
    {
      int red = enum_thorough_simple.red;
      enum_thorough_simple.colourTest1(red);
      enum_thorough_simple.colourTest2(red);
      enum_thorough_simple.colourTest3(red);
      enum_thorough_simple.colourTest4(red);
      enum_thorough_simple.setMyColour(red);
    }
    {
      SpeedClass s = new SpeedClass();
      int speed = SpeedClass.slow;
      if (s.speedTest1(speed) != speed) throw new RuntimeException("speedTest 1 failed");
      if (s.speedTest2(speed) != speed) throw new RuntimeException("speedTest 2 failed");
      if (s.speedTest3(speed) != speed) throw new RuntimeException("speedTest 3 failed");
      if (s.speedTest4(speed) != speed) throw new RuntimeException("speedTest 4 failed");
      if (s.speedTest5(speed) != speed) throw new RuntimeException("speedTest 5 failed");
      if (s.speedTest6(speed) != speed) throw new RuntimeException("speedTest 6 failed");
      if (s.speedTest7(speed) != speed) throw new RuntimeException("speedTest 7 failed");
      if (s.speedTest8(speed) != speed) throw new RuntimeException("speedTest 8 failed");

      if (enum_thorough_simple.speedTest1(speed) != speed) throw new RuntimeException("speedTest Global 1 failed");
      if (enum_thorough_simple.speedTest2(speed) != speed) throw new RuntimeException("speedTest Global 2 failed");
      if (enum_thorough_simple.speedTest3(speed) != speed) throw new RuntimeException("speedTest Global 3 failed");
      if (enum_thorough_simple.speedTest4(speed) != speed) throw new RuntimeException("speedTest Global 4 failed");
      if (enum_thorough_simple.speedTest5(speed) != speed) throw new RuntimeException("speedTest Global 5 failed");
    }
    {
      SpeedClass s = new SpeedClass();
      int slow = SpeedClass.slow;
      int lightning = SpeedClass.lightning;

      if (s.getMySpeedtd1() != slow) throw new RuntimeException("mySpeedtd1 1 failed");
      if (s.getMySpeedtd1() != 10) throw new RuntimeException("mySpeedtd1 2 failed");

      s.setMySpeedtd1(lightning);
      if (s.getMySpeedtd1() != lightning) throw new RuntimeException("mySpeedtd1 3 failed");
      if (s.getMySpeedtd1() != 31) throw new RuntimeException("mySpeedtd1 4 failed");
    }
    {
      if (enum_thorough_simple.namedanonTest1(enum_thorough_simple.NamedAnon2) != enum_thorough_simple.NamedAnon2) throw new RuntimeException("namedanonTest 1 failed");
    }
    {
      int val = enum_thorough_simple.TwoNames2;
      if (enum_thorough_simple.twonamesTest1(val) != val) throw new RuntimeException("twonamesTest 1 failed");
      if (enum_thorough_simple.twonamesTest2(val) != val) throw new RuntimeException("twonamesTest 2 failed");
      if (enum_thorough_simple.twonamesTest3(val) != val) throw new RuntimeException("twonamesTest 3 failed");
    }
    {
      TwoNamesStruct t = new TwoNamesStruct();
      int val = TwoNamesStruct.TwoNamesStruct1;
      if (t.twonamesTest1(val) != val) throw new RuntimeException("twonamesTest 1 failed");
      if (t.twonamesTest2(val) != val) throw new RuntimeException("twonamesTest 2 failed");
      if (t.twonamesTest3(val) != val) throw new RuntimeException("twonamesTest 3 failed");
    }
    {
      int val = enum_thorough_simple.NamedAnonSpace2;
      if (enum_thorough_simple.namedanonspaceTest1(val) != val) throw new RuntimeException("namedanonspaceTest 1 failed");
      if (enum_thorough_simple.namedanonspaceTest2(val) != val) throw new RuntimeException("namedanonspaceTest 2 failed");
      if (enum_thorough_simple.namedanonspaceTest3(val) != val) throw new RuntimeException("namedanonspaceTest 3 failed");
      if (enum_thorough_simple.namedanonspaceTest4(val) != val) throw new RuntimeException("namedanonspaceTest 4 failed");
    }
    {
      TemplateClassInt t = new TemplateClassInt();
      int galileo = TemplateClassInt.galileo;

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

      if (enum_thorough_simple.scientistsTest1(galileo) != galileo) throw new RuntimeException("scientistsTest Global 1 failed");
      if (enum_thorough_simple.scientistsTest2(galileo) != galileo) throw new RuntimeException("scientistsTest Global 2 failed");
      if (enum_thorough_simple.scientistsTest3(galileo) != galileo) throw new RuntimeException("scientistsTest Global 3 failed");
      if (enum_thorough_simple.scientistsTest4(galileo) != galileo) throw new RuntimeException("scientistsTest Global 4 failed");
      if (enum_thorough_simple.scientistsTest5(galileo) != galileo) throw new RuntimeException("scientistsTest Global 5 failed");
      if (enum_thorough_simple.scientistsTest6(galileo) != galileo) throw new RuntimeException("scientistsTest Global 6 failed");
      if (enum_thorough_simple.scientistsTest7(galileo) != galileo) throw new RuntimeException("scientistsTest Global 7 failed");
      if (enum_thorough_simple.scientistsTest8(galileo) != galileo) throw new RuntimeException("scientistsTest Global 8 failed");
    }
    {
      TClassInt t = new TClassInt();
      int bell = TClassInt.bell;
      int galileo = TemplateClassInt.galileo;
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

      if (enum_thorough_simple.scientistsNameTest1(bell) != bell) throw new RuntimeException("scientistsNameTest Global 1 failed");
      if (enum_thorough_simple.scientistsNameTest2(bell) != bell) throw new RuntimeException("scientistsNameTest Global 2 failed");
      if (enum_thorough_simple.scientistsNameTest3(bell) != bell) throw new RuntimeException("scientistsNameTest Global 3 failed");
      if (enum_thorough_simple.scientistsNameTest4(bell) != bell) throw new RuntimeException("scientistsNameTest Global 4 failed");
      if (enum_thorough_simple.scientistsNameTest5(bell) != bell) throw new RuntimeException("scientistsNameTest Global 5 failed");
      if (enum_thorough_simple.scientistsNameTest6(bell) != bell) throw new RuntimeException("scientistsNameTest Global 6 failed");
      if (enum_thorough_simple.scientistsNameTest7(bell) != bell) throw new RuntimeException("scientistsNameTest Global 7 failed");

      if (enum_thorough_simple.scientistsNameSpaceTest1(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 1 failed");
      if (enum_thorough_simple.scientistsNameSpaceTest2(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 2 failed");
      if (enum_thorough_simple.scientistsNameSpaceTest3(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 3 failed");
      if (enum_thorough_simple.scientistsNameSpaceTest4(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 4 failed");
      if (enum_thorough_simple.scientistsNameSpaceTest5(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 5 failed");
      if (enum_thorough_simple.scientistsNameSpaceTest6(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 6 failed");
      if (enum_thorough_simple.scientistsNameSpaceTest7(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 7 failed");

      if (enum_thorough_simple.scientistsNameSpaceTest8(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 8 failed");
      if (enum_thorough_simple.scientistsNameSpaceTest9(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 9 failed");
      if (enum_thorough_simple.scientistsNameSpaceTestA(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global A failed");
      if (enum_thorough_simple.scientistsNameSpaceTestB(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global B failed");
      if (enum_thorough_simple.scientistsNameSpaceTestC(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global C failed");
      if (enum_thorough_simple.scientistsNameSpaceTestD(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global D failed");
      if (enum_thorough_simple.scientistsNameSpaceTestE(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global E failed");

      if (enum_thorough_simple.scientistsNameSpaceTestF(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global F failed");
      if (enum_thorough_simple.scientistsNameSpaceTestG(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global G failed");
      if (enum_thorough_simple.scientistsNameSpaceTestH(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global H failed");
      if (enum_thorough_simple.scientistsNameSpaceTestI(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global I failed");
      if (enum_thorough_simple.scientistsNameSpaceTestJ(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global J failed");
      if (enum_thorough_simple.scientistsNameSpaceTestK(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global K failed");
      if (enum_thorough_simple.scientistsNameSpaceTestL(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global L failed");
    }
    {
      int val = enum_thorough_simple.argh;
      if (enum_thorough_simple.renameTest1(val) != val) throw new RuntimeException("renameTest Global 1 failed");
      if (enum_thorough_simple.renameTest2(val) != val) throw new RuntimeException("renameTest Global 2 failed");
    }
    {
      NewNameStruct n = new NewNameStruct();
      if (n.renameTest1(NewNameStruct.bang) != NewNameStruct.bang) throw new RuntimeException("renameTest 1 failed");
      if (n.renameTest2(NewNameStruct.bang) != NewNameStruct.bang) throw new RuntimeException("renameTest 2 failed");
      if (n.renameTest3(NewNameStruct.simple1) != NewNameStruct.simple1) throw new RuntimeException("renameTest 3 failed");
      if (n.renameTest4(NewNameStruct.doublename1) != NewNameStruct.doublename1) throw new RuntimeException("renameTest 4 failed");
      if (n.renameTest5(NewNameStruct.doublename1) != NewNameStruct.doublename1) throw new RuntimeException("renameTest 5 failed");
      if (n.renameTest6(NewNameStruct.singlename1) != NewNameStruct.singlename1) throw new RuntimeException("renameTest 6 failed");
    }
    {
      if (enum_thorough_simple.renameTest3(NewNameStruct.bang) != NewNameStruct.bang) throw new RuntimeException("renameTest Global 3 failed");
      if (enum_thorough_simple.renameTest4(NewNameStruct.simple1) != NewNameStruct.simple1) throw new RuntimeException("renameTest Global 4 failed");
      if (enum_thorough_simple.renameTest5(NewNameStruct.doublename1) != NewNameStruct.doublename1) throw new RuntimeException("renameTest Global 5 failed");
      if (enum_thorough_simple.renameTest6(NewNameStruct.doublename1) != NewNameStruct.doublename1) throw new RuntimeException("renameTest Global 6 failed");
      if (enum_thorough_simple.renameTest7(NewNameStruct.singlename1) != NewNameStruct.singlename1) throw new RuntimeException("renameTest Global 7 failed");
    }
    {
      TreesClass t = new TreesClass();
      int pine = TreesClass.pine;

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

      if (enum_thorough_simple.treesTest1(pine) != pine) throw new RuntimeException("treesTest Global 1 failed");
      if (enum_thorough_simple.treesTest2(pine) != pine) throw new RuntimeException("treesTest Global 2 failed");
      if (enum_thorough_simple.treesTest3(pine) != pine) throw new RuntimeException("treesTest Global 3 failed");
      if (enum_thorough_simple.treesTest4(pine) != pine) throw new RuntimeException("treesTest Global 4 failed");
      if (enum_thorough_simple.treesTest5(pine) != pine) throw new RuntimeException("treesTest Global 5 failed");
      if (enum_thorough_simple.treesTest6(pine) != pine) throw new RuntimeException("treesTest Global 6 failed");
      if (enum_thorough_simple.treesTest7(pine) != pine) throw new RuntimeException("treesTest Global 7 failed");
      if (enum_thorough_simple.treesTest8(pine) != pine) throw new RuntimeException("treesTest Global 8 failed");
      if (enum_thorough_simple.treesTest9(pine) != pine) throw new RuntimeException("treesTest Global 9 failed");
      if (enum_thorough_simple.treesTestA(pine) != pine) throw new RuntimeException("treesTest Global A failed");
      if (enum_thorough_simple.treesTestB(pine) != pine) throw new RuntimeException("treesTest Global B failed");
      if (enum_thorough_simple.treesTestC(pine) != pine) throw new RuntimeException("treesTest Global C failed");
      if (enum_thorough_simple.treesTestD(pine) != pine) throw new RuntimeException("treesTest Global D failed");
      if (enum_thorough_simple.treesTestE(pine) != pine) throw new RuntimeException("treesTest Global E failed");
      if (enum_thorough_simple.treesTestF(pine) != pine) throw new RuntimeException("treesTest Global F failed");
      if (enum_thorough_simple.treesTestG(pine) != pine) throw new RuntimeException("treesTest Global G failed");
      if (enum_thorough_simple.treesTestH(pine) != pine) throw new RuntimeException("treesTest Global H failed");
      if (enum_thorough_simple.treesTestI(pine) != pine) throw new RuntimeException("treesTest Global I failed");
      if (enum_thorough_simple.treesTestJ(pine) != pine) throw new RuntimeException("treesTest Global J failed");
      if (enum_thorough_simple.treesTestK(pine) != pine) throw new RuntimeException("treesTest Global K failed");
      if (enum_thorough_simple.treesTestL(pine) != pine) throw new RuntimeException("treesTest Global L failed");
      if (enum_thorough_simple.treesTestM(pine) != pine) throw new RuntimeException("treesTest Global M failed");
//      if (enum_thorough_simple.treesTestN(pine) != pine) throw new RuntimeException("treesTest Global N failed");
      if (enum_thorough_simple.treesTestO(pine) != pine) throw new RuntimeException("treesTest Global O failed");
      if (enum_thorough_simple.treesTestP(pine) != pine) throw new RuntimeException("treesTest Global P failed");
      if (enum_thorough_simple.treesTestQ(pine) != pine) throw new RuntimeException("treesTest Global Q failed");
      if (enum_thorough_simple.treesTestR(pine) != pine) throw new RuntimeException("treesTest Global R failed");
    }
    {
      HairStruct h = new HairStruct();
      int ginger = HairStruct.ginger;

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

      int red = enum_thorough_simple.red;
      if (h.colourTest1(red) != red) throw new RuntimeException("colourTest HairStruct 1 failed");
      if (h.colourTest2(red) != red) throw new RuntimeException("colourTest HairStruct 2 failed");
      if (h.namedanonTest1(enum_thorough_simple.NamedAnon2) != enum_thorough_simple.NamedAnon2) throw new RuntimeException("namedanonTest HairStruct 1 failed");
      if (h.namedanonspaceTest1(enum_thorough_simple.NamedAnonSpace2) != enum_thorough_simple.NamedAnonSpace2) throw new RuntimeException("namedanonspaceTest HairStruct 1 failed");

      int fir = TreesClass.fir;
      if (h.treesGlobalTest1(fir) != fir) throw new RuntimeException("treesGlobalTest1 HairStruct 1 failed");
      if (h.treesGlobalTest2(fir) != fir) throw new RuntimeException("treesGlobalTest1 HairStruct 2 failed");
      if (h.treesGlobalTest3(fir) != fir) throw new RuntimeException("treesGlobalTest1 HairStruct 3 failed");
      if (h.treesGlobalTest4(fir) != fir) throw new RuntimeException("treesGlobalTest1 HairStruct 4 failed");
    }
    {
      int blonde = HairStruct.blonde;
      if (enum_thorough_simple.hairTest1(blonde) != blonde) throw new RuntimeException("hairTest Global 1 failed");
      if (enum_thorough_simple.hairTest2(blonde) != blonde) throw new RuntimeException("hairTest Global 2 failed");
      if (enum_thorough_simple.hairTest3(blonde) != blonde) throw new RuntimeException("hairTest Global 3 failed");
      if (enum_thorough_simple.hairTest4(blonde) != blonde) throw new RuntimeException("hairTest Global 4 failed");
      if (enum_thorough_simple.hairTest5(blonde) != blonde) throw new RuntimeException("hairTest Global 5 failed");
      if (enum_thorough_simple.hairTest6(blonde) != blonde) throw new RuntimeException("hairTest Global 6 failed");
      if (enum_thorough_simple.hairTest7(blonde) != blonde) throw new RuntimeException("hairTest Global 7 failed");
      if (enum_thorough_simple.hairTest8(blonde) != blonde) throw new RuntimeException("hairTest Global 8 failed");
      if (enum_thorough_simple.hairTest9(blonde) != blonde) throw new RuntimeException("hairTest Global 9 failed");
      if (enum_thorough_simple.hairTestA(blonde) != blonde) throw new RuntimeException("hairTest Global A failed");
      if (enum_thorough_simple.hairTestB(blonde) != blonde) throw new RuntimeException("hairTest Global B failed");
      if (enum_thorough_simple.hairTestC(blonde) != blonde) throw new RuntimeException("hairTest Global C failed");

      if (enum_thorough_simple.hairTestA1(blonde) != blonde) throw new RuntimeException("hairTest Global A1 failed");
      if (enum_thorough_simple.hairTestA2(blonde) != blonde) throw new RuntimeException("hairTest Global A2 failed");
      if (enum_thorough_simple.hairTestA3(blonde) != blonde) throw new RuntimeException("hairTest Global A3 failed");
      if (enum_thorough_simple.hairTestA4(blonde) != blonde) throw new RuntimeException("hairTest Global A4 failed");
      if (enum_thorough_simple.hairTestA5(blonde) != blonde) throw new RuntimeException("hairTest Global A5 failed");
      if (enum_thorough_simple.hairTestA6(blonde) != blonde) throw new RuntimeException("hairTest Global A6 failed");
      if (enum_thorough_simple.hairTestA7(blonde) != blonde) throw new RuntimeException("hairTest Global A7 failed");
      if (enum_thorough_simple.hairTestA8(blonde) != blonde) throw new RuntimeException("hairTest Global A8 failed");
      if (enum_thorough_simple.hairTestA9(blonde) != blonde) throw new RuntimeException("hairTest Global A9 failed");
      if (enum_thorough_simple.hairTestAA(blonde) != blonde) throw new RuntimeException("hairTest Global AA failed");
      if (enum_thorough_simple.hairTestAB(blonde) != blonde) throw new RuntimeException("hairTest Global AB failed");
      if (enum_thorough_simple.hairTestAC(blonde) != blonde) throw new RuntimeException("hairTest Global AC failed");

      if (enum_thorough_simple.hairTestB1(blonde) != blonde) throw new RuntimeException("hairTest Global B1 failed");
      if (enum_thorough_simple.hairTestB2(blonde) != blonde) throw new RuntimeException("hairTest Global B2 failed");
      if (enum_thorough_simple.hairTestB3(blonde) != blonde) throw new RuntimeException("hairTest Global B3 failed");
      if (enum_thorough_simple.hairTestB4(blonde) != blonde) throw new RuntimeException("hairTest Global B4 failed");
      if (enum_thorough_simple.hairTestB5(blonde) != blonde) throw new RuntimeException("hairTest Global B5 failed");
      if (enum_thorough_simple.hairTestB6(blonde) != blonde) throw new RuntimeException("hairTest Global B6 failed");
      if (enum_thorough_simple.hairTestB7(blonde) != blonde) throw new RuntimeException("hairTest Global B7 failed");
      if (enum_thorough_simple.hairTestB8(blonde) != blonde) throw new RuntimeException("hairTest Global B8 failed");
      if (enum_thorough_simple.hairTestB9(blonde) != blonde) throw new RuntimeException("hairTest Global B9 failed");
      if (enum_thorough_simple.hairTestBA(blonde) != blonde) throw new RuntimeException("hairTest Global BA failed");
      if (enum_thorough_simple.hairTestBB(blonde) != blonde) throw new RuntimeException("hairTest Global BB failed");
      if (enum_thorough_simple.hairTestBC(blonde) != blonde) throw new RuntimeException("hairTest Global BC failed");

      if (enum_thorough_simple.hairTestC1(blonde) != blonde) throw new RuntimeException("hairTest Global C1 failed");
      if (enum_thorough_simple.hairTestC2(blonde) != blonde) throw new RuntimeException("hairTest Global C2 failed");
      if (enum_thorough_simple.hairTestC3(blonde) != blonde) throw new RuntimeException("hairTest Global C3 failed");
      if (enum_thorough_simple.hairTestC4(blonde) != blonde) throw new RuntimeException("hairTest Global C4 failed");
      if (enum_thorough_simple.hairTestC5(blonde) != blonde) throw new RuntimeException("hairTest Global C5 failed");
      if (enum_thorough_simple.hairTestC6(blonde) != blonde) throw new RuntimeException("hairTest Global C6 failed");
      if (enum_thorough_simple.hairTestC7(blonde) != blonde) throw new RuntimeException("hairTest Global C7 failed");
      if (enum_thorough_simple.hairTestC8(blonde) != blonde) throw new RuntimeException("hairTest Global C8 failed");
      if (enum_thorough_simple.hairTestC9(blonde) != blonde) throw new RuntimeException("hairTest Global C9 failed");
      if (enum_thorough_simple.hairTestCA(blonde) != blonde) throw new RuntimeException("hairTest Global CA failed");
      if (enum_thorough_simple.hairTestCB(blonde) != blonde) throw new RuntimeException("hairTest Global CB failed");
      if (enum_thorough_simple.hairTestCC(blonde) != blonde) throw new RuntimeException("hairTest Global CC failed");
    }
    {
      FirStruct f = new FirStruct();
      int blonde = HairStruct.blonde;

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
      enum_thorough_simple.setGlobalInstance(enum_thorough_simple.globalinstance2);
      if (enum_thorough_simple.getGlobalInstance() != enum_thorough_simple.globalinstance2) throw new RuntimeException("GlobalInstance 1 failed");

      Instances i = new Instances();
      i.setMemberInstance(Instances.memberinstance3);
      if (i.getMemberInstance() != Instances.memberinstance3) throw new RuntimeException("MemberInstance 1 failed");
    }
    // ignore enum item tests start
    {
      if (enum_thorough_simple.ignoreATest(IgnoreTest.ignoreA_zero) != 0) throw new RuntimeException("ignoreATest 0 failed");
      if (enum_thorough_simple.ignoreATest(IgnoreTest.ignoreA_three) != 3) throw new RuntimeException("ignoreATest 3 failed");
      if (enum_thorough_simple.ignoreATest(IgnoreTest.ignoreA_ten) != 10) throw new RuntimeException("ignoreATest 10 failed");
      if (enum_thorough_simple.ignoreATest(IgnoreTest.ignoreA_eleven) != 11) throw new RuntimeException("ignoreATest 11 failed");
      if (enum_thorough_simple.ignoreATest(IgnoreTest.ignoreA_thirteen) != 13) throw new RuntimeException("ignoreATest 13 failed");
      if (enum_thorough_simple.ignoreATest(IgnoreTest.ignoreA_fourteen) != 14) throw new RuntimeException("ignoreATest 14 failed");
      if (enum_thorough_simple.ignoreATest(IgnoreTest.ignoreA_twenty) != 20) throw new RuntimeException("ignoreATest 20 failed");
      if (enum_thorough_simple.ignoreATest(IgnoreTest.ignoreA_thirty) != 30) throw new RuntimeException("ignoreATest 30 failed");
      if (enum_thorough_simple.ignoreATest(IgnoreTest.ignoreA_thirty_two) != 32) throw new RuntimeException("ignoreATest 32 failed");
      if (enum_thorough_simple.ignoreATest(IgnoreTest.ignoreA_thirty_three) != 33) throw new RuntimeException("ignoreATest 33 failed");
    }
    {
      if (enum_thorough_simple.ignoreBTest(IgnoreTest.ignoreB_eleven) != 11) throw new RuntimeException("ignoreBTest 11 failed");
      if (enum_thorough_simple.ignoreBTest(IgnoreTest.ignoreB_twelve) != 12) throw new RuntimeException("ignoreBTest 12 failed");
      if (enum_thorough_simple.ignoreBTest(IgnoreTest.ignoreB_thirty_one) != 31) throw new RuntimeException("ignoreBTest 31 failed");
      if (enum_thorough_simple.ignoreBTest(IgnoreTest.ignoreB_thirty_two) != 32) throw new RuntimeException("ignoreBTest 32 failed");
      if (enum_thorough_simple.ignoreBTest(IgnoreTest.ignoreB_forty_one) != 41) throw new RuntimeException("ignoreBTest 41 failed");
      if (enum_thorough_simple.ignoreBTest(IgnoreTest.ignoreB_forty_two) != 42) throw new RuntimeException("ignoreBTest 42 failed");
    }
    {
      if (enum_thorough_simple.ignoreCTest(IgnoreTest.ignoreC_ten) != 10) throw new RuntimeException("ignoreCTest 10 failed");
      if (enum_thorough_simple.ignoreCTest(IgnoreTest.ignoreC_twelve) != 12) throw new RuntimeException("ignoreCTest 12 failed");
      if (enum_thorough_simple.ignoreCTest(IgnoreTest.ignoreC_thirty) != 30) throw new RuntimeException("ignoreCTest 30 failed");
      if (enum_thorough_simple.ignoreCTest(IgnoreTest.ignoreC_thirty_two) != 32) throw new RuntimeException("ignoreCTest 32 failed");
      if (enum_thorough_simple.ignoreCTest(IgnoreTest.ignoreC_forty) != 40) throw new RuntimeException("ignoreCTest 40 failed");
      if (enum_thorough_simple.ignoreCTest(IgnoreTest.ignoreC_forty_two) != 42) throw new RuntimeException("ignoreCTest 42 failed");
    }
    {
      if (enum_thorough_simple.ignoreDTest(IgnoreTest.ignoreD_twenty_one) != 21) throw new RuntimeException("ignoreDTest 21 failed");
      if (enum_thorough_simple.ignoreDTest(IgnoreTest.ignoreD_twenty_two) != 22) throw new RuntimeException("ignoreDTest 22 failed");
    }
    {
      if (enum_thorough_simple.ignoreETest(IgnoreTest.ignoreE_zero) != 0) throw new RuntimeException("ignoreETest 0 failed");
      if (enum_thorough_simple.ignoreETest(IgnoreTest.ignoreE_twenty_one) != 21) throw new RuntimeException("ignoreETest 21 failed");
      if (enum_thorough_simple.ignoreETest(IgnoreTest.ignoreE_twenty_two) != 22) throw new RuntimeException("ignoreETest 22 failed");
    }
    // ignore enum item tests end
    {
      if (enum_thorough_simple.repeatTest(enum_thorough_simpleConstants.one) != 1) throw new RuntimeException("repeatTest 1 failed");
      if (enum_thorough_simple.repeatTest(enum_thorough_simpleConstants.initial) != 1) throw new RuntimeException("repeatTest 2 failed");
      if (enum_thorough_simple.repeatTest(enum_thorough_simpleConstants.two) != 2) throw new RuntimeException("repeatTest 3 failed");
      if (enum_thorough_simple.repeatTest(enum_thorough_simpleConstants.three) != 3) throw new RuntimeException("repeatTest 4 failed");
      if (enum_thorough_simple.repeatTest(enum_thorough_simpleConstants.llast) != 3) throw new RuntimeException("repeatTest 5 failed");
      if (enum_thorough_simple.repeatTest(enum_thorough_simpleConstants.end) != 3) throw new RuntimeException("repeatTest 6 failed");
    }
    // different types
    {
      if (enum_thorough_simple.differentTypesTest(enum_thorough_simpleConstants.typeint) != 10) throw new RuntimeException("differentTypes 1 failed");
      if (enum_thorough_simple.differentTypesTest(enum_thorough_simpleConstants.typeboolfalse) != 0) throw new RuntimeException("differentTypes 2 failed");
      if (enum_thorough_simple.differentTypesTest(enum_thorough_simpleConstants.typebooltrue) != 1) throw new RuntimeException("differentTypes 3 failed");
      if (enum_thorough_simple.differentTypesTest(enum_thorough_simpleConstants.typebooltwo) != 2) throw new RuntimeException("differentTypes 4 failed");
      if (enum_thorough_simple.differentTypesTest(enum_thorough_simpleConstants.typechar) != 'C') throw new RuntimeException("differentTypes 5 failed");
      if (enum_thorough_simple.differentTypesTest(enum_thorough_simpleConstants.typedefaultint) != 'D') throw new RuntimeException("differentTypes 6 failed");

      int global_enum = enum_thorough_simple.global_typeint;
      if (enum_thorough_simple.globalDifferentTypesTest(global_enum) != 10) throw new RuntimeException("global differentTypes 1 failed");
      global_enum = enum_thorough_simple.global_typeboolfalse;
      if (enum_thorough_simple.globalDifferentTypesTest(global_enum) != 0) throw new RuntimeException("global differentTypes 2 failed");
      global_enum = enum_thorough_simple.global_typebooltrue;
      if (enum_thorough_simple.globalDifferentTypesTest(global_enum) != 1) throw new RuntimeException("global differentTypes 3 failed");
      global_enum = enum_thorough_simple.global_typebooltwo;
      if (enum_thorough_simple.globalDifferentTypesTest(global_enum) != 2) throw new RuntimeException("global differentTypes 4 failed");
      global_enum = enum_thorough_simple.global_typechar;
      if (enum_thorough_simple.globalDifferentTypesTest(global_enum) != 'C') throw new RuntimeException("global differentTypes 5 failed");
      global_enum = enum_thorough_simple.global_typedefaultint;
      if (enum_thorough_simple.globalDifferentTypesTest(global_enum) != 'D') throw new RuntimeException("global differentTypes 6 failed");
    }
  }
}

