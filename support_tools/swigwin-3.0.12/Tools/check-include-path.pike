/**
 * This is a helper script to identify the proper include path
 * for Pike header files. It should be run with the full path
 * to the Pike executable as its single argument, e.g.
 *
 *     pike check-include-path.pike /usr/local/bin/pike
 *
 * and its output should be the correct path to the header
 * files, e.g.
 *
 *     /usr/local/pike/7.2.239/include/pike
 *
 */

int main(int argc, array(string) argv)
{
    string prefix = replace(argv[1], "/bin/pike", "");
    write(prefix + "/pike/" + __MAJOR__ + "." + __MINOR__ + "." + __BUILD__ + "/include/pike");
    return 0;
}
