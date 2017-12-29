Port on OpenVMS 7.3 using CC 6.5 and CXX 6.5


Building procedure:
$ @logicals
$ @build_all

the logicals swig_root is defined by the procedure logicals.com.
The logicals.com procedure can be invoke with an optional argument
for the define command, for example:
$ @logicals "/system/exec"


genbuild.py is the python program use to generate all the procedures in the
[vms.scripts] directory.


jf.pieronne@laposte.net
