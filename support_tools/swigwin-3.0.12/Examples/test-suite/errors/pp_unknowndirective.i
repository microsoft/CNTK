%module xxx

/* Regression test for bug introduced in 3.0.4 and fixed in 3.0.6 - the '%std'
 * here led to SWIG calling abort().
 */
%typemap(jstype) std::vector<std::string>, const %std::vector<std::string>&, std::vector<std::string>&  "List<String>"

/* This used to give the rather cryptic "Syntax error in input(1)." prior to
 * SWIG 3.0.4.  This testcase checks that the improved message is actually
 * issued.
 */
%remane("typo") tyop;
