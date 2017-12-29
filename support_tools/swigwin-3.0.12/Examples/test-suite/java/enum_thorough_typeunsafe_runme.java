
import enum_thorough_typeunsafe.*;

public class enum_thorough_typeunsafe_runme {

  static {
    try {
        System.loadLibrary("enum_thorough_typeunsafe");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    {
      // Anonymous enums
      int i = enum_thorough_typeunsafe.AnonEnum1;
      if (enum_thorough_typeunsafe.ReallyAnInteger != 200) throw new RuntimeException("Test Anon 1 failed");
      int j = enum_thorough_typeunsafe.AnonSpaceEnum1;
      int k = AnonStruct.AnonStructEnum1;
    }
    {
      int red = colour.red;
      enum_thorough_typeunsafe.colourTest1(red);
      enum_thorough_typeunsafe.colourTest2(red);
      enum_thorough_typeunsafe.colourTest3(red);
      enum_thorough_typeunsafe.colourTest4(red);
      enum_thorough_typeunsafe.setMyColour(red);
    }
    {
      SpeedClass s = new SpeedClass();
      int speed = SpeedClass.speed.slow;
      if (s.speedTest1(speed) != speed) throw new RuntimeException("speedTest 1 failed");
      if (s.speedTest2(speed) != speed) throw new RuntimeException("speedTest 2 failed");
      if (s.speedTest3(speed) != speed) throw new RuntimeException("speedTest 3 failed");
      if (s.speedTest4(speed) != speed) throw new RuntimeException("speedTest 4 failed");
      if (s.speedTest5(speed) != speed) throw new RuntimeException("speedTest 5 failed");
      if (s.speedTest6(speed) != speed) throw new RuntimeException("speedTest 6 failed");
      if (s.speedTest7(speed) != speed) throw new RuntimeException("speedTest 7 failed");
      if (s.speedTest8(speed) != speed) throw new RuntimeException("speedTest 8 failed");

      if (enum_thorough_typeunsafe.speedTest1(speed) != speed) throw new RuntimeException("speedTest Global 1 failed");
      if (enum_thorough_typeunsafe.speedTest2(speed) != speed) throw new RuntimeException("speedTest Global 2 failed");
      if (enum_thorough_typeunsafe.speedTest3(speed) != speed) throw new RuntimeException("speedTest Global 3 failed");
      if (enum_thorough_typeunsafe.speedTest4(speed) != speed) throw new RuntimeException("speedTest Global 4 failed");
      if (enum_thorough_typeunsafe.speedTest5(speed) != speed) throw new RuntimeException("speedTest Global 5 failed");
    }
    {
      SpeedClass s = new SpeedClass();
      int slow = SpeedClass.speed.slow;
      int lightning = SpeedClass.speed.lightning;

      if (s.getMySpeedtd1() != slow) throw new RuntimeException("mySpeedtd1 1 failed");
      if (s.getMySpeedtd1() != 10) throw new RuntimeException("mySpeedtd1 2 failed");

      s.setMySpeedtd1(lightning);
      if (s.getMySpeedtd1() != lightning) throw new RuntimeException("mySpeedtd1 3 failed");
      if (s.getMySpeedtd1() != 31) throw new RuntimeException("mySpeedtd1 4 failed");
    }
    {
      if (enum_thorough_typeunsafe.namedanonTest1(namedanon.NamedAnon2) != namedanon.NamedAnon2) throw new RuntimeException("namedanonTest 1 failed");
    }
    {
      int val = twonames.TwoNames2;
      if (enum_thorough_typeunsafe.twonamesTest1(val) != val) throw new RuntimeException("twonamesTest 1 failed");
      if (enum_thorough_typeunsafe.twonamesTest2(val) != val) throw new RuntimeException("twonamesTest 2 failed");
      if (enum_thorough_typeunsafe.twonamesTest3(val) != val) throw new RuntimeException("twonamesTest 3 failed");
    }
    {
      TwoNamesStruct t = new TwoNamesStruct();
      int val = TwoNamesStruct.twonames.TwoNamesStruct1;
      if (t.twonamesTest1(val) != val) throw new RuntimeException("twonamesTest 1 failed");
      if (t.twonamesTest2(val) != val) throw new RuntimeException("twonamesTest 2 failed");
      if (t.twonamesTest3(val) != val) throw new RuntimeException("twonamesTest 3 failed");
    }
    {
      int val = namedanonspace.NamedAnonSpace2;
      if (enum_thorough_typeunsafe.namedanonspaceTest1(val) != val) throw new RuntimeException("namedanonspaceTest 1 failed");
      if (enum_thorough_typeunsafe.namedanonspaceTest2(val) != val) throw new RuntimeException("namedanonspaceTest 2 failed");
      if (enum_thorough_typeunsafe.namedanonspaceTest3(val) != val) throw new RuntimeException("namedanonspaceTest 3 failed");
      if (enum_thorough_typeunsafe.namedanonspaceTest4(val) != val) throw new RuntimeException("namedanonspaceTest 4 failed");
    }
    {
      TemplateClassInt t = new TemplateClassInt();
      int galileo = TemplateClassInt.scientists.galileo;

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

      if (enum_thorough_typeunsafe.scientistsTest1(galileo) != galileo) throw new RuntimeException("scientistsTest Global 1 failed");
      if (enum_thorough_typeunsafe.scientistsTest2(galileo) != galileo) throw new RuntimeException("scientistsTest Global 2 failed");
      if (enum_thorough_typeunsafe.scientistsTest3(galileo) != galileo) throw new RuntimeException("scientistsTest Global 3 failed");
      if (enum_thorough_typeunsafe.scientistsTest4(galileo) != galileo) throw new RuntimeException("scientistsTest Global 4 failed");
      if (enum_thorough_typeunsafe.scientistsTest5(galileo) != galileo) throw new RuntimeException("scientistsTest Global 5 failed");
      if (enum_thorough_typeunsafe.scientistsTest6(galileo) != galileo) throw new RuntimeException("scientistsTest Global 6 failed");
      if (enum_thorough_typeunsafe.scientistsTest7(galileo) != galileo) throw new RuntimeException("scientistsTest Global 7 failed");
      if (enum_thorough_typeunsafe.scientistsTest8(galileo) != galileo) throw new RuntimeException("scientistsTest Global 8 failed");
    }
    {
      TClassInt t = new TClassInt();
      int bell = TClassInt.scientists.bell;
      int galileo = TemplateClassInt.scientists.galileo;
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

      if (enum_thorough_typeunsafe.scientistsNameTest1(bell) != bell) throw new RuntimeException("scientistsNameTest Global 1 failed");
      if (enum_thorough_typeunsafe.scientistsNameTest2(bell) != bell) throw new RuntimeException("scientistsNameTest Global 2 failed");
      if (enum_thorough_typeunsafe.scientistsNameTest3(bell) != bell) throw new RuntimeException("scientistsNameTest Global 3 failed");
      if (enum_thorough_typeunsafe.scientistsNameTest4(bell) != bell) throw new RuntimeException("scientistsNameTest Global 4 failed");
      if (enum_thorough_typeunsafe.scientistsNameTest5(bell) != bell) throw new RuntimeException("scientistsNameTest Global 5 failed");
      if (enum_thorough_typeunsafe.scientistsNameTest6(bell) != bell) throw new RuntimeException("scientistsNameTest Global 6 failed");
      if (enum_thorough_typeunsafe.scientistsNameTest7(bell) != bell) throw new RuntimeException("scientistsNameTest Global 7 failed");

      if (enum_thorough_typeunsafe.scientistsNameSpaceTest1(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 1 failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTest2(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 2 failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTest3(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 3 failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTest4(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 4 failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTest5(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 5 failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTest6(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 6 failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTest7(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 7 failed");

      if (enum_thorough_typeunsafe.scientistsNameSpaceTest8(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 8 failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTest9(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global 9 failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTestA(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global A failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTestB(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global B failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTestC(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global C failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTestD(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global D failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTestE(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global E failed");

      if (enum_thorough_typeunsafe.scientistsNameSpaceTestF(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global F failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTestG(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global G failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTestH(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global H failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTestI(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global I failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTestJ(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global J failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTestK(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global K failed");
      if (enum_thorough_typeunsafe.scientistsNameSpaceTestL(bell) != bell) throw new RuntimeException("scientistsNameSpaceTest Global L failed");
    }
    {
      int val = newname.argh;
      if (enum_thorough_typeunsafe.renameTest1(val) != val) throw new RuntimeException("renameTest Global 1 failed");
      if (enum_thorough_typeunsafe.renameTest2(val) != val) throw new RuntimeException("renameTest Global 2 failed");
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
      if (enum_thorough_typeunsafe.renameTest3(NewNameStruct.enumeration.bang) != NewNameStruct.enumeration.bang) throw new RuntimeException("renameTest Global 3 failed");
      if (enum_thorough_typeunsafe.renameTest4(NewNameStruct.simplerenamed.simple1) != NewNameStruct.simplerenamed.simple1) throw new RuntimeException("renameTest Global 4 failed");
      if (enum_thorough_typeunsafe.renameTest5(NewNameStruct.doublenamerenamed.doublename1) != NewNameStruct.doublenamerenamed.doublename1) throw new RuntimeException("renameTest Global 5 failed");
      if (enum_thorough_typeunsafe.renameTest6(NewNameStruct.doublenamerenamed.doublename1) != NewNameStruct.doublenamerenamed.doublename1) throw new RuntimeException("renameTest Global 6 failed");
      if (enum_thorough_typeunsafe.renameTest7(NewNameStruct.singlenamerenamed.singlename1) != NewNameStruct.singlenamerenamed.singlename1) throw new RuntimeException("renameTest Global 7 failed");
    }
    {
      TreesClass t = new TreesClass();
      int pine = TreesClass.trees.pine;

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

      if (enum_thorough_typeunsafe.treesTest1(pine) != pine) throw new RuntimeException("treesTest Global 1 failed");
      if (enum_thorough_typeunsafe.treesTest2(pine) != pine) throw new RuntimeException("treesTest Global 2 failed");
      if (enum_thorough_typeunsafe.treesTest3(pine) != pine) throw new RuntimeException("treesTest Global 3 failed");
      if (enum_thorough_typeunsafe.treesTest4(pine) != pine) throw new RuntimeException("treesTest Global 4 failed");
      if (enum_thorough_typeunsafe.treesTest5(pine) != pine) throw new RuntimeException("treesTest Global 5 failed");
      if (enum_thorough_typeunsafe.treesTest6(pine) != pine) throw new RuntimeException("treesTest Global 6 failed");
      if (enum_thorough_typeunsafe.treesTest7(pine) != pine) throw new RuntimeException("treesTest Global 7 failed");
      if (enum_thorough_typeunsafe.treesTest8(pine) != pine) throw new RuntimeException("treesTest Global 8 failed");
      if (enum_thorough_typeunsafe.treesTest9(pine) != pine) throw new RuntimeException("treesTest Global 9 failed");
      if (enum_thorough_typeunsafe.treesTestA(pine) != pine) throw new RuntimeException("treesTest Global A failed");
      if (enum_thorough_typeunsafe.treesTestB(pine) != pine) throw new RuntimeException("treesTest Global B failed");
      if (enum_thorough_typeunsafe.treesTestC(pine) != pine) throw new RuntimeException("treesTest Global C failed");
      if (enum_thorough_typeunsafe.treesTestD(pine) != pine) throw new RuntimeException("treesTest Global D failed");
      if (enum_thorough_typeunsafe.treesTestE(pine) != pine) throw new RuntimeException("treesTest Global E failed");
      if (enum_thorough_typeunsafe.treesTestF(pine) != pine) throw new RuntimeException("treesTest Global F failed");
      if (enum_thorough_typeunsafe.treesTestG(pine) != pine) throw new RuntimeException("treesTest Global G failed");
      if (enum_thorough_typeunsafe.treesTestH(pine) != pine) throw new RuntimeException("treesTest Global H failed");
      if (enum_thorough_typeunsafe.treesTestI(pine) != pine) throw new RuntimeException("treesTest Global I failed");
      if (enum_thorough_typeunsafe.treesTestJ(pine) != pine) throw new RuntimeException("treesTest Global J failed");
      if (enum_thorough_typeunsafe.treesTestK(pine) != pine) throw new RuntimeException("treesTest Global K failed");
      if (enum_thorough_typeunsafe.treesTestL(pine) != pine) throw new RuntimeException("treesTest Global L failed");
      if (enum_thorough_typeunsafe.treesTestM(pine) != pine) throw new RuntimeException("treesTest Global M failed");
//      if (enum_thorough_typeunsafe.treesTestN(pine) != pine) throw new RuntimeException("treesTest Global N failed");
      if (enum_thorough_typeunsafe.treesTestO(pine) != pine) throw new RuntimeException("treesTest Global O failed");
      if (enum_thorough_typeunsafe.treesTestP(pine) != pine) throw new RuntimeException("treesTest Global P failed");
      if (enum_thorough_typeunsafe.treesTestQ(pine) != pine) throw new RuntimeException("treesTest Global Q failed");
      if (enum_thorough_typeunsafe.treesTestR(pine) != pine) throw new RuntimeException("treesTest Global R failed");
    }
    {
      HairStruct h = new HairStruct();
      int ginger = HairStruct.hair.ginger;

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

      int red = colour.red;
      if (h.colourTest1(red) != red) throw new RuntimeException("colourTest HairStruct 1 failed");
      if (h.colourTest2(red) != red) throw new RuntimeException("colourTest HairStruct 2 failed");
      if (h.namedanonTest1(namedanon.NamedAnon2) != namedanon.NamedAnon2) throw new RuntimeException("namedanonTest HairStruct 1 failed");
      if (h.namedanonspaceTest1(namedanonspace.NamedAnonSpace2) != namedanonspace.NamedAnonSpace2) throw new RuntimeException("namedanonspaceTest HairStruct 1 failed");

      int fir = TreesClass.trees.fir;
      if (h.treesGlobalTest1(fir) != fir) throw new RuntimeException("treesGlobalTest1 HairStruct 1 failed");
      if (h.treesGlobalTest2(fir) != fir) throw new RuntimeException("treesGlobalTest1 HairStruct 2 failed");
      if (h.treesGlobalTest3(fir) != fir) throw new RuntimeException("treesGlobalTest1 HairStruct 3 failed");
      if (h.treesGlobalTest4(fir) != fir) throw new RuntimeException("treesGlobalTest1 HairStruct 4 failed");
    }
    {
      int blonde = HairStruct.hair.blonde;
      if (enum_thorough_typeunsafe.hairTest1(blonde) != blonde) throw new RuntimeException("hairTest Global 1 failed");
      if (enum_thorough_typeunsafe.hairTest2(blonde) != blonde) throw new RuntimeException("hairTest Global 2 failed");
      if (enum_thorough_typeunsafe.hairTest3(blonde) != blonde) throw new RuntimeException("hairTest Global 3 failed");
      if (enum_thorough_typeunsafe.hairTest4(blonde) != blonde) throw new RuntimeException("hairTest Global 4 failed");
      if (enum_thorough_typeunsafe.hairTest5(blonde) != blonde) throw new RuntimeException("hairTest Global 5 failed");
      if (enum_thorough_typeunsafe.hairTest6(blonde) != blonde) throw new RuntimeException("hairTest Global 6 failed");
      if (enum_thorough_typeunsafe.hairTest7(blonde) != blonde) throw new RuntimeException("hairTest Global 7 failed");
      if (enum_thorough_typeunsafe.hairTest8(blonde) != blonde) throw new RuntimeException("hairTest Global 8 failed");
      if (enum_thorough_typeunsafe.hairTest9(blonde) != blonde) throw new RuntimeException("hairTest Global 9 failed");
      if (enum_thorough_typeunsafe.hairTestA(blonde) != blonde) throw new RuntimeException("hairTest Global A failed");
      if (enum_thorough_typeunsafe.hairTestB(blonde) != blonde) throw new RuntimeException("hairTest Global B failed");
      if (enum_thorough_typeunsafe.hairTestC(blonde) != blonde) throw new RuntimeException("hairTest Global C failed");

      if (enum_thorough_typeunsafe.hairTestA1(blonde) != blonde) throw new RuntimeException("hairTest Global A1 failed");
      if (enum_thorough_typeunsafe.hairTestA2(blonde) != blonde) throw new RuntimeException("hairTest Global A2 failed");
      if (enum_thorough_typeunsafe.hairTestA3(blonde) != blonde) throw new RuntimeException("hairTest Global A3 failed");
      if (enum_thorough_typeunsafe.hairTestA4(blonde) != blonde) throw new RuntimeException("hairTest Global A4 failed");
      if (enum_thorough_typeunsafe.hairTestA5(blonde) != blonde) throw new RuntimeException("hairTest Global A5 failed");
      if (enum_thorough_typeunsafe.hairTestA6(blonde) != blonde) throw new RuntimeException("hairTest Global A6 failed");
      if (enum_thorough_typeunsafe.hairTestA7(blonde) != blonde) throw new RuntimeException("hairTest Global A7 failed");
      if (enum_thorough_typeunsafe.hairTestA8(blonde) != blonde) throw new RuntimeException("hairTest Global A8 failed");
      if (enum_thorough_typeunsafe.hairTestA9(blonde) != blonde) throw new RuntimeException("hairTest Global A9 failed");
      if (enum_thorough_typeunsafe.hairTestAA(blonde) != blonde) throw new RuntimeException("hairTest Global AA failed");
      if (enum_thorough_typeunsafe.hairTestAB(blonde) != blonde) throw new RuntimeException("hairTest Global AB failed");
      if (enum_thorough_typeunsafe.hairTestAC(blonde) != blonde) throw new RuntimeException("hairTest Global AC failed");

      if (enum_thorough_typeunsafe.hairTestB1(blonde) != blonde) throw new RuntimeException("hairTest Global B1 failed");
      if (enum_thorough_typeunsafe.hairTestB2(blonde) != blonde) throw new RuntimeException("hairTest Global B2 failed");
      if (enum_thorough_typeunsafe.hairTestB3(blonde) != blonde) throw new RuntimeException("hairTest Global B3 failed");
      if (enum_thorough_typeunsafe.hairTestB4(blonde) != blonde) throw new RuntimeException("hairTest Global B4 failed");
      if (enum_thorough_typeunsafe.hairTestB5(blonde) != blonde) throw new RuntimeException("hairTest Global B5 failed");
      if (enum_thorough_typeunsafe.hairTestB6(blonde) != blonde) throw new RuntimeException("hairTest Global B6 failed");
      if (enum_thorough_typeunsafe.hairTestB7(blonde) != blonde) throw new RuntimeException("hairTest Global B7 failed");
      if (enum_thorough_typeunsafe.hairTestB8(blonde) != blonde) throw new RuntimeException("hairTest Global B8 failed");
      if (enum_thorough_typeunsafe.hairTestB9(blonde) != blonde) throw new RuntimeException("hairTest Global B9 failed");
      if (enum_thorough_typeunsafe.hairTestBA(blonde) != blonde) throw new RuntimeException("hairTest Global BA failed");
      if (enum_thorough_typeunsafe.hairTestBB(blonde) != blonde) throw new RuntimeException("hairTest Global BB failed");
      if (enum_thorough_typeunsafe.hairTestBC(blonde) != blonde) throw new RuntimeException("hairTest Global BC failed");

      if (enum_thorough_typeunsafe.hairTestC1(blonde) != blonde) throw new RuntimeException("hairTest Global C1 failed");
      if (enum_thorough_typeunsafe.hairTestC2(blonde) != blonde) throw new RuntimeException("hairTest Global C2 failed");
      if (enum_thorough_typeunsafe.hairTestC3(blonde) != blonde) throw new RuntimeException("hairTest Global C3 failed");
      if (enum_thorough_typeunsafe.hairTestC4(blonde) != blonde) throw new RuntimeException("hairTest Global C4 failed");
      if (enum_thorough_typeunsafe.hairTestC5(blonde) != blonde) throw new RuntimeException("hairTest Global C5 failed");
      if (enum_thorough_typeunsafe.hairTestC6(blonde) != blonde) throw new RuntimeException("hairTest Global C6 failed");
      if (enum_thorough_typeunsafe.hairTestC7(blonde) != blonde) throw new RuntimeException("hairTest Global C7 failed");
      if (enum_thorough_typeunsafe.hairTestC8(blonde) != blonde) throw new RuntimeException("hairTest Global C8 failed");
      if (enum_thorough_typeunsafe.hairTestC9(blonde) != blonde) throw new RuntimeException("hairTest Global C9 failed");
      if (enum_thorough_typeunsafe.hairTestCA(blonde) != blonde) throw new RuntimeException("hairTest Global CA failed");
      if (enum_thorough_typeunsafe.hairTestCB(blonde) != blonde) throw new RuntimeException("hairTest Global CB failed");
      if (enum_thorough_typeunsafe.hairTestCC(blonde) != blonde) throw new RuntimeException("hairTest Global CC failed");
    }
    {
      FirStruct f = new FirStruct();
      int blonde = HairStruct.hair.blonde;

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
      enum_thorough_typeunsafe.setGlobalInstance(enum_thorough_typeunsafe.globalinstance2);
      if (enum_thorough_typeunsafe.getGlobalInstance() != enum_thorough_typeunsafe.globalinstance2) throw new RuntimeException("GlobalInstance 1 failed");

      Instances i = new Instances();
      i.setMemberInstance(Instances.memberinstance3);
      if (i.getMemberInstance() != Instances.memberinstance3) throw new RuntimeException("MemberInstance 1 failed");
    }
    // ignore enum item tests start
    {
      if (enum_thorough_typeunsafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_zero) != 0) throw new RuntimeException("ignoreATest 0 failed");
      if (enum_thorough_typeunsafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_three) != 3) throw new RuntimeException("ignoreATest 3 failed");
      if (enum_thorough_typeunsafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_ten) != 10) throw new RuntimeException("ignoreATest 10 failed");
      if (enum_thorough_typeunsafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_eleven) != 11) throw new RuntimeException("ignoreATest 11 failed");
      if (enum_thorough_typeunsafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_thirteen) != 13) throw new RuntimeException("ignoreATest 13 failed");
      if (enum_thorough_typeunsafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_fourteen) != 14) throw new RuntimeException("ignoreATest 14 failed");
      if (enum_thorough_typeunsafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_twenty) != 20) throw new RuntimeException("ignoreATest 20 failed");
      if (enum_thorough_typeunsafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_thirty) != 30) throw new RuntimeException("ignoreATest 30 failed");
      if (enum_thorough_typeunsafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_thirty_two) != 32) throw new RuntimeException("ignoreATest 32 failed");
      if (enum_thorough_typeunsafe.ignoreATest(IgnoreTest.IgnoreA.ignoreA_thirty_three) != 33) throw new RuntimeException("ignoreATest 33 failed");
    }
    {
      if (enum_thorough_typeunsafe.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_eleven) != 11) throw new RuntimeException("ignoreBTest 11 failed");
      if (enum_thorough_typeunsafe.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_twelve) != 12) throw new RuntimeException("ignoreBTest 12 failed");
      if (enum_thorough_typeunsafe.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_thirty_one) != 31) throw new RuntimeException("ignoreBTest 31 failed");
      if (enum_thorough_typeunsafe.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_thirty_two) != 32) throw new RuntimeException("ignoreBTest 32 failed");
      if (enum_thorough_typeunsafe.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_forty_one) != 41) throw new RuntimeException("ignoreBTest 41 failed");
      if (enum_thorough_typeunsafe.ignoreBTest(IgnoreTest.IgnoreB.ignoreB_forty_two) != 42) throw new RuntimeException("ignoreBTest 42 failed");
    }
    {
      if (enum_thorough_typeunsafe.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_ten) != 10) throw new RuntimeException("ignoreCTest 10 failed");
      if (enum_thorough_typeunsafe.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_twelve) != 12) throw new RuntimeException("ignoreCTest 12 failed");
      if (enum_thorough_typeunsafe.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_thirty) != 30) throw new RuntimeException("ignoreCTest 30 failed");
      if (enum_thorough_typeunsafe.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_thirty_two) != 32) throw new RuntimeException("ignoreCTest 32 failed");
      if (enum_thorough_typeunsafe.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_forty) != 40) throw new RuntimeException("ignoreCTest 40 failed");
      if (enum_thorough_typeunsafe.ignoreCTest(IgnoreTest.IgnoreC.ignoreC_forty_two) != 42) throw new RuntimeException("ignoreCTest 42 failed");
    }
    {
      if (enum_thorough_typeunsafe.ignoreDTest(IgnoreTest.IgnoreD.ignoreD_twenty_one) != 21) throw new RuntimeException("ignoreDTest 21 failed");
      if (enum_thorough_typeunsafe.ignoreDTest(IgnoreTest.IgnoreD.ignoreD_twenty_two) != 22) throw new RuntimeException("ignoreDTest 22 failed");
    }
    {
      if (enum_thorough_typeunsafe.ignoreETest(IgnoreTest.IgnoreE.ignoreE_zero) != 0) throw new RuntimeException("ignoreETest 0 failed");
      if (enum_thorough_typeunsafe.ignoreETest(IgnoreTest.IgnoreE.ignoreE_twenty_one) != 21) throw new RuntimeException("ignoreETest 21 failed");
      if (enum_thorough_typeunsafe.ignoreETest(IgnoreTest.IgnoreE.ignoreE_twenty_two) != 22) throw new RuntimeException("ignoreETest 22 failed");
    }
    // ignore enum item tests end
    {
      if (enum_thorough_typeunsafe.repeatTest(repeat.one) != 1) throw new RuntimeException("repeatTest 1 failed");
      if (enum_thorough_typeunsafe.repeatTest(repeat.initial) != 1) throw new RuntimeException("repeatTest 2 failed");
      if (enum_thorough_typeunsafe.repeatTest(repeat.two) != 2) throw new RuntimeException("repeatTest 3 failed");
      if (enum_thorough_typeunsafe.repeatTest(repeat.three) != 3) throw new RuntimeException("repeatTest 4 failed");
      if (enum_thorough_typeunsafe.repeatTest(repeat.llast) != 3) throw new RuntimeException("repeatTest 5 failed");
      if (enum_thorough_typeunsafe.repeatTest(repeat.end) != 3) throw new RuntimeException("repeatTest 6 failed");
    }
    // different types
    {
      if (enum_thorough_typeunsafe.differentTypesTest(DifferentTypes.typeint) != 10) throw new RuntimeException("differentTypes 1 failed");
      if (enum_thorough_typeunsafe.differentTypesTest(DifferentTypes.typeboolfalse) != 0) throw new RuntimeException("differentTypes 2 failed");
      if (enum_thorough_typeunsafe.differentTypesTest(DifferentTypes.typebooltrue) != 1) throw new RuntimeException("differentTypes 3 failed");
      if (enum_thorough_typeunsafe.differentTypesTest(DifferentTypes.typebooltwo) != 2) throw new RuntimeException("differentTypes 4 failed");
      if (enum_thorough_typeunsafe.differentTypesTest(DifferentTypes.typechar) != 'C') throw new RuntimeException("differentTypes 5 failed");
      if (enum_thorough_typeunsafe.differentTypesTest(DifferentTypes.typedefaultint) != 'D') throw new RuntimeException("differentTypes 6 failed");

      int global_enum = enum_thorough_typeunsafe.global_typeint;
      if (enum_thorough_typeunsafe.globalDifferentTypesTest(global_enum) != 10) throw new RuntimeException("global differentTypes 1 failed");
      global_enum = enum_thorough_typeunsafe.global_typeboolfalse;
      if (enum_thorough_typeunsafe.globalDifferentTypesTest(global_enum) != 0) throw new RuntimeException("global differentTypes 2 failed");
      global_enum = enum_thorough_typeunsafe.global_typebooltrue;
      if (enum_thorough_typeunsafe.globalDifferentTypesTest(global_enum) != 1) throw new RuntimeException("global differentTypes 3 failed");
      global_enum = enum_thorough_typeunsafe.global_typebooltwo;
      if (enum_thorough_typeunsafe.globalDifferentTypesTest(global_enum) != 2) throw new RuntimeException("global differentTypes 4 failed");
      global_enum = enum_thorough_typeunsafe.global_typechar;
      if (enum_thorough_typeunsafe.globalDifferentTypesTest(global_enum) != 'C') throw new RuntimeException("global differentTypes 5 failed");
      global_enum = enum_thorough_typeunsafe.global_typedefaultint;
      if (enum_thorough_typeunsafe.globalDifferentTypesTest(global_enum) != 'D') throw new RuntimeException("global differentTypes 6 failed");
    }
  }
}

