/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * swigopt.h
 *
 * Header file for the SWIG command line processing functions
 * ----------------------------------------------------------------------------- */

 extern void  Swig_init_args(int argc, char **argv);
 extern void  Swig_mark_arg(int n);
 extern int   Swig_check_marked(int n);
 extern void  Swig_check_options(int check_input);
 extern void  Swig_arg_error(void);
