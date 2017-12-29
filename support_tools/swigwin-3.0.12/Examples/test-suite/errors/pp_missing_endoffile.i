%module xxx
/* %beginfile and %endoffile are internal directives inserted when %include is
 * used.  Users should never use them directly, but test coverage for this
 * error message still seems useful to have.
 */
%includefile "dummy.i" %beginfile

