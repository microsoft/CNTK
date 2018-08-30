dnl @synopsis KLM_SYS_WEAK_ALIAS
dnl @author Kevin L. Mitchell <klmitch@mit.edu>
dnl @version $Id: klm_sys_weak_alias.m4,v 1.1 2006/09/27 02:53:33 guidod Exp $
dnl
dnl Determines whether weak aliases are supported on the system, and
dnl if so, what scheme is used to declare them.  Also checks to see if
dnl aliases can cross object file boundaries, as some systems don't
dnl permit them to.
dnl
dnl Most systems permit something called a "weak alias" or "weak
dnl symbol."  These aliases permit a library to provide a stub form of
dnl a routine defined in another library, thus allowing the first
dnl library to operate even if the other library is not linked.  This
dnl macro will check for support of weak aliases, figure out what
dnl schemes are available, and determine some characteristics of the
dnl weak alias support--primarily, whether a weak alias declared in
dnl one object file may be referenced from another object file.
dnl
dnl There are four known schemes of declaring weak symbols; each
dnl scheme is checked in turn, and the first one found is prefered.
dnl Note that only one of the mentioned preprocessor macros will be
dnl defined!
dnl
dnl 1. Function attributes
dnl
dnl    This scheme was first introduced by the GNU C compiler, and
dnl    attaches attributes to particular functions.  It is among the
dnl    easiest to use, and so is the first one checked.  If this
dnl    scheme is detected, the preprocessor macro
dnl    HAVE_SYS_WEAK_ALIAS_ATTRIBUTE will be defined to 1.  This
dnl    scheme is used as in the following code fragment:
dnl
dnl	 void __weakf(int c)
dnl	 {
dnl	   /* Function definition... */
dnl	 }
dnl
dnl	 void weakf(int c) __attribute__((weak, alias("__weakf")));
dnl
dnl 2. #pragma weak
dnl
dnl    This scheme is in use by many compilers other than the GNU C
dnl    compiler.  It is also particularly easy to use, and fairly
dnl    portable--well, as portable as these things get.  If this
dnl    scheme is detected first, the preprocessor macro
dnl    HAVE_SYS_WEAK_ALIAS_PRAGMA will be defined to 1.  This scheme
dnl    is used as in the following code fragment:
dnl
dnl	 extern void weakf(int c);
dnl	 #pragma weak weakf = __weakf
dnl	 void __weakf(int c)
dnl	 {
dnl	   /* Function definition... */
dnl	 }
dnl
dnl 3. #pragma _HP_SECONDARY_DEF
dnl
dnl    This scheme appears to be in use by the HP compiler.  As it is
dnl    rather specialized, this is one of the last schemes checked.
dnl    If it is the first one detected, the preprocessor macro
dnl    HAVE_SYS_WEAK_ALIAS_HPSECONDARY will be defined to 1.  This
dnl    scheme is used as in the following code fragment:
dnl
dnl	 extern void weakf(int c);
dnl	 #pragma _HP_SECONDARY_DEF __weakf weakf
dnl	 void __weakf(int c)
dnl	 {
dnl	   /* Function definition... */
dnl	 }
dnl
dnl 4. #pragma _CRI duplicate
dnl
dnl    This scheme appears to be in use by the Cray compiler.  As it
dnl    is rather specialized, it too is one of the last schemes
dnl    checked.  If it is the first one detected, the preprocessor
dnl    macro HAVE_SYS_WEAK_ALIAS_CRIDUPLICATE will be defined to 1.
dnl    This scheme is used as in the following code fragment:
dnl
dnl	 extern void weakf(int c);
dnl	 #pragma _CRI duplicate weakf as __weakf
dnl	 void __weakf(int c)
dnl	 {
dnl	   /* Function definition... */
dnl	 }
dnl
dnl In addition to the preprocessor macros listed above, if any scheme
dnl is found, the preprocessor macro HAVE_SYS_WEAK_ALIAS will also be
dnl defined to 1.
dnl
dnl Once a weak aliasing scheme has been found, a check will be
dnl performed to see if weak aliases are honored across object file
dnl boundaries.  If they are, the HAVE_SYS_WEAK_ALIAS_CROSSFILE
dnl preprocessor macro is defined to 1.
dnl
dnl This Autoconf macro also makes two substitutions.  The first,
dnl WEAK_ALIAS, contains the name of the scheme found (one of
dnl "attribute", "pragma", "hpsecondary", or "criduplicate"), or "no"
dnl if no weak aliasing scheme was found.  The second,
dnl WEAK_ALIAS_CROSSFILE, is set to "yes" or "no" depending on whether
dnl or not weak aliases may cross object file boundaries.
dnl
AC_DEFUN([KLM_SYS_WEAK_ALIAS], [
  # starting point: no aliasing scheme yet...
  klm_sys_weak_alias=no

  # Figure out what kind of aliasing may be supported...
  _KLM_SYS_WEAK_ALIAS_ATTRIBUTE
  _KLM_SYS_WEAK_ALIAS_PRAGMA
  _KLM_SYS_WEAK_ALIAS_HPSECONDARY
  _KLM_SYS_WEAK_ALIAS_CRIDUPLICATE

  # Do we actually support aliasing?
  AC_CACHE_CHECK([how to create weak aliases with $CC],
		 [klm_cv_sys_weak_alias],
		 [klm_cv_sys_weak_alias=$klm_sys_weak_alias])

  # OK, set a #define
  AS_IF([test $klm_cv_sys_weak_alias != no], [
    AC_DEFINE([HAVE_SYS_WEAK_ALIAS], 1,
	      [Define this if your system can create weak aliases])
  ])

  # Can aliases cross object file boundaries?
  _KLM_SYS_WEAK_ALIAS_CROSSFILE

  # OK, remember the results
  AC_SUBST([WEAK_ALIAS], [$klm_cv_sys_weak_alias])
  AC_SUBST([WEAK_ALIAS_CROSSFILE], [$klm_cv_sys_weak_alias_crossfile])
])

AC_DEFUN([_KLM_SYS_WEAK_ALIAS_ATTRIBUTE],
[ # Test whether compiler accepts __attribute__ form of weak aliasing
  AC_CACHE_CHECK([whether $CC accepts function __attribute__((weak,alias()))],
  [klm_cv_sys_weak_alias_attribute], [
    # We add -Werror if it's gcc to force an error exit if the weak attribute
    # isn't understood
    AS_IF([test $GCC = yes], [
      save_CFLAGS=$CFLAGS
      CFLAGS=-Werror])

    # Try linking with a weak alias...
    AC_LINK_IFELSE([
      AC_LANG_PROGRAM([
void __weakf(int c) {}
void weakf(int c) __attribute__((weak, alias("__weakf")));],
	[weakf(0)])],
      [klm_cv_sys_weak_alias_attribute=yes],
      [klm_cv_sys_weak_alias_attribute=no])

    # Restore original CFLAGS
    AS_IF([test $GCC = yes], [
      CFLAGS=$save_CFLAGS])
  ])

  # What was the result of the test?
  AS_IF([test $klm_sys_weak_alias = no &&
	 test $klm_cv_sys_weak_alias_attribute = yes], [
    klm_sys_weak_alias=attribute
    AC_DEFINE([HAVE_SYS_WEAK_ALIAS_ATTRIBUTE], 1,
	      [Define this if weak aliases may be created with __attribute__])
  ])
])

AC_DEFUN([_KLM_SYS_WEAK_ALIAS_PRAGMA],
[ # Test whether compiler accepts #pragma form of weak aliasing
  AC_CACHE_CHECK([whether $CC supports @%:@pragma weak],
  [klm_cv_sys_weak_alias_pragma], [

    # Try linking with a weak alias...
    AC_LINK_IFELSE([
      AC_LANG_PROGRAM([
extern void weakf(int c);
@%:@pragma weak weakf = __weakf
void __weakf(int c) {}],
	[weakf(0)])],
      [klm_cv_sys_weak_alias_pragma=yes],
      [klm_cv_sys_weak_alias_pragma=no])
  ])

  # What was the result of the test?
  AS_IF([ test $klm_cv_sys_weak_alias_pragma = yes], [
    klm_sys_weak_alias=pragma
    AC_DEFINE([HAVE_SYS_WEAK_ALIAS_PRAGMA], 1,
	      [Define this if weak aliases may be created with @%:@pragma weak])
  ])
])

AC_DEFUN([_KLM_SYS_WEAK_ALIAS_HPSECONDARY],
[ # Test whether compiler accepts _HP_SECONDARY_DEF pragma from HP...
  AC_CACHE_CHECK([whether $CC supports @%:@pragma _HP_SECONDARY_DEF],
  [klm_cv_sys_weak_alias_hpsecondary], [

    # Try linking with a weak alias...
    AC_LINK_IFELSE([
      AC_LANG_PROGRAM([
extern void weakf(int c);
@%:@pragma _HP_SECONDARY_DEF __weakf weakf
void __weakf(int c) {}],
	[weakf(0)])],
      [klm_cv_sys_weak_alias_hpsecondary=yes],
      [klm_cv_sys_weak_alias_hpsecondary=no])
  ])

  # What was the result of the test?
  AS_IF([test $klm_sys_weak_alias = no &&
	 test $klm_cv_sys_weak_alias_hpsecondary = yes], [
    klm_sys_weak_alias=hpsecondary
    AC_DEFINE([HAVE_SYS_WEAK_ALIAS_HPSECONDARY], 1,
	      [Define this if weak aliases may be created with @%:@pragma _HP_SECONDARY_DEF])
  ])
])

AC_DEFUN([_KLM_SYS_WEAK_ALIAS_CRIDUPLICATE],
[ # Test whether compiler accepts "_CRI duplicate" pragma from Cray
  AC_CACHE_CHECK([whether $CC supports @%:@pragma _CRI duplicate],
  [klm_cv_sys_weak_alias_criduplicate], [

    # Try linking with a weak alias...
    AC_LINK_IFELSE([
      AC_LANG_PROGRAM([
extern void weakf(int c);
@%:@pragma _CRI duplicate weakf as __weakf
void __weakf(int c) {}],
	[weakf(0)])],
      [klm_cv_sys_weak_alias_criduplicate=yes],
      [klm_cv_sys_weak_alias_criduplicate=no])
  ])

  # What was the result of the test?
  AS_IF([test $klm_sys_weak_alias = no &&
	 test $klm_cv_sys_weak_alias_criduplicate = yes], [
    klm_sys_weak_alias=criduplicate
    AC_DEFINE([HAVE_SYS_WEAK_ALIAS_CRIDUPLICATE], 1,
	      [Define this if weak aliases may be created with @%:@pragma _CRI duplicate])
  ])
])

dnl Note: This macro is modeled closely on AC_LINK_IFELSE, and in fact
dnl depends on some implementation details of that macro, particularly
dnl its use of _AC_MSG_LOG_CONFTEST to log the failed test program and
dnl its use of ac_link for running the linker.
AC_DEFUN([_KLM_SYS_WEAK_ALIAS_CROSSFILE],
[ # Check to see if weak aliases can cross object file boundaries
  AC_CACHE_CHECK([whether $CC supports weak aliases across object file boundaries],
  [klm_cv_sys_weak_alias_crossfile], [
    AS_IF([test $klm_cv_sys_weak_alias = no],
	  [klm_cv_sys_weak_alias_crossfile=no], [
dnl Must build our own test files...
      # conftest1 contains our weak alias definition...
      cat >conftest1.$ac_ext <<_ACEOF
/* confdefs.h.  */
_ACEOF
      cat confdefs.h >>conftest1.$ac_ext
      cat >>conftest1.$ac_ext <<_ACEOF
/* end confdefs.h.  */

@%:@ifndef HAVE_SYS_WEAK_ALIAS_ATTRIBUTE
extern void weakf(int c);
@%:@endif
@%:@if defined(HAVE_SYS_WEAK_ALIAS_PRAGMA)
@%:@pragma weak weakf = __weakf
@%:@elif defined(HAVE_SYS_WEAK_ALIAS_HPSECONDARY)
@%:@pragma _HP_SECONDARY_DEF __weakf weakf
@%:@elif defined(HAVE_SYS_WEAK_ALIAS_CRIDUPLICATE)
@%:@pragma _CRI duplicate weakf as __weakf
@%:@endif
void __weakf(int c) {}
@%:@ifdef HAVE_SYS_WEAK_ALIAS_ATTRIBUTE
void weakf(int c) __attribute((weak, alias("__weakf")));
@%:@endif
_ACEOF
      # And conftest2 contains our main routine that calls it
      cat >conftest2.$ac_ext <<_ACEOF
/* confdefs.h.  */
_ACEOF
      cat confdefs.h >> conftest2.$ac_ext
      cat >>conftest2.$ac_ext <<_ACEOF
/* end confdefs.h.  */

extern void weakf(int c);
int
main ()
{
  weakf(0);
  return 0;
}
_ACEOF
      # We must remove the object files (if any) ourselves...
      rm -f conftest2.$ac_objext conftest$ac_exeext

      # Change ac_link to compile *2* files together
      save_aclink=$ac_link
      ac_link=`echo "$ac_link" | \
	       sed -e 's/conftest\(\.\$ac_ext\)/conftest1\1 conftest2\1/'`
dnl Substitute our own routine for logging the conftest
m4_pushdef([_AC_MSG_LOG_CONFTEST],
[echo "$as_me: failed program was:" >&AS_MESSAGE_LOG_FD
echo ">>> conftest1.$ac_ext" >&AS_MESSAGE_LOG_FD
sed "s/^/| /" conftest1.$ac_ext >&AS_MESSAGE_LOG_FD
echo ">>> conftest2.$ac_ext" >&AS_MESSAGE_LOG_FD
sed "s/^/| /" conftest2.$ac_ext >&AS_MESSAGE_LOG_FD
])dnl
      # Since we created the files ourselves, don't use SOURCE argument
      AC_LINK_IFELSE(, [klm_cv_sys_weak_alias_crossfile=yes],
		     [klm_cv_sys_weak_alias_crossfile=no])
dnl Restore _AC_MSG_LOG_CONFTEST
m4_popdef([_AC_MSG_LOG_CONFTEST])dnl
      # Restore ac_link
      ac_link=$save_aclink

      # We must remove the object files (if any) and C files ourselves...
      rm -f conftest1.$ac_ext conftest2.$ac_ext \
	    conftest1.$ac_objext conftest2.$ac_objext
    ])
  ])

  # What were the results of the test?
  AS_IF([test $klm_cv_sys_weak_alias_crossfile = yes], [
    AC_DEFINE([HAVE_SYS_WEAK_ALIAS_CROSSFILE], 1,
	      [Define this if weak aliases in other files are honored])
  ])
])
