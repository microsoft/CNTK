int main()
{
  write("ICONST  = %d (should be 42)\n", .example.ICONST);
  write("FCONST  = %f (should be 2.1828)\n", .example.FCONST);
  write("CCONST  = %c (should be 'x')\n", .example.CCONST);
  write("CCONST2 = %c (this should be on a new line)\n", .example.CCONST2);
  write("SCONST  = %s (should be 'Hello World')\n", .example.SCONST);
  write("SCONST2 = %s (should be '\"Hello World\"')\n", .example.SCONST2);
  write("EXPR    = %f (should be 48.5484)\n", .example.EXPR);
  write("iconst  = %d (should be 37)\n", .example.iconst);
  write("fconst  = %f (should be 3.14)\n", .example.fconst);

  if (search(indices(.example), "EXTERN") == -1)
    write("EXTERN isn't defined (good)\n");
  else
    write("EXTERN is defined (bad)\n");

  if (search(indices(.example), "FOO") == -1)
    write("FOO isn't defined (good)\n");
  else
    write("FOO is defined (bad)\n");

  return 0;
}
