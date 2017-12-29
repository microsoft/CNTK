import preproc_line_file.*;

public class preproc_line_file_runme {

  static {
    try {
        System.loadLibrary("preproc_line_file");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  private static void test_file(String file, String suffix) throws Throwable
  {
// For swig-3.0.1 and earlier
//      String FILENAME_WINDOWS = "Examples\\test-suite\\preproc_line_file.i";
//      String FILENAME_UNIX = "Examples/test-suite/preproc_line_file.i";

      String FILENAME_WINDOWS2 = "Examples\\test-suite\\java\\..\\preproc_line_file.i";
      String FILENAME_UNIX2 = "Examples/test-suite/java/../preproc_line_file.i";

      String FILENAME_WINDOWS3 = "..\\.\\..\\preproc_line_file.i";
      String FILENAME_UNIX3 = ".././../preproc_line_file.i";

      // We don't test for exact equality here because the file names are relative to the build directory, which can be different from the source directory,
      // under Unix. But they do need to end with the same path components.
      if (!file.endsWith(FILENAME_UNIX2 + suffix) && !file.endsWith(FILENAME_WINDOWS2 + suffix) &&
          !file.endsWith(FILENAME_UNIX3 + suffix) && !file.endsWith(FILENAME_WINDOWS3 + suffix))
          throw new RuntimeException("file \"" + file + "\" doesn't end with " + FILENAME_UNIX2 + suffix + " or " + FILENAME_UNIX3 + suffix);
  }

  public static void main(String argv[]) throws Throwable
  {
    int myline = preproc_line_file.MYLINE;
    int myline_adjusted = preproc_line_file.MYLINE_ADJUSTED;
    if (myline != 4)
      throw new RuntimeException("preproc failure");
    if (myline + 100 + 1 != myline_adjusted)
      throw new RuntimeException("preproc failure");

    test_file(preproc_line_file.MYFILE, "");
    test_file(preproc_line_file.MYFILE_ADJUSTED, ".bak");

    if (!preproc_line_file.MY_STRINGNUM_A.equals("my15"))
      throw new RuntimeException("preproc failed MY_STRINGNUM_A");

    if (!preproc_line_file.MY_STRINGNUM_B.equals("my16"))
      throw new RuntimeException("preproc failed MY_STRINGNUM_B");

    if (preproc_line_file.getThing27() != -1)
      throw new RuntimeException("preproc failure");

    if (preproc_line_file.getThing28() != -2)
      throw new RuntimeException("preproc failure");

    if (preproc_line_file.MYLINE2 != 30)
      throw new RuntimeException("preproc failure");

    if (SillyStruct.LINE_NUMBER != 52)
      throw new RuntimeException("preproc failure");

    if (SillyMacroClass.LINE_NUM != 56)
      throw new RuntimeException("preproc failure");

    if (SillyMulMacroStruc.LINE_NUM != 81)
      throw new RuntimeException("preproc failure");

    if (preproc_line_file.INLINE_LINE != 87)
      throw new RuntimeException("preproc failure");

    test_file(preproc_line_file.INLINE_FILE, "");

    if (Slash.LINE_NUM != 93)
      throw new RuntimeException("preproc failure");

  }
}
