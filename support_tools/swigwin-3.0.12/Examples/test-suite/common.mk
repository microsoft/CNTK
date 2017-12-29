#######################################################################
# SWIG test suite makefile.
# The test suite comprises many different test cases, which have
# typically produced bugs in the past. The aim is to have the test 
# cases compiling for every language modules. Some testcase have
# a runtime test which is written in each of the module's language.
#
# This makefile runs SWIG on the testcases, compiles the c/c++ code
# then builds the object code for use by the language.
# To complete a test in a language follow these guidelines: 
# 1) Add testcases to CPP_TEST_CASES (c++) or C_TEST_CASES (c) or
#    MULTI_CPP_TEST_CASES (multi-module c++ tests)
# 2) If not already done, create a makefile which:
#    a) Defines LANGUAGE matching a language rule in Examples/Makefile, 
#       for example LANGUAGE = java
#    b) Define rules for %.ctest, %.cpptest, %.multicpptest and %.clean.
#    c) Define srcdir, top_srcdir and top_builddir (these are the
#       equivalent to configure's variables of the same name).
# 3) One off special commandline options for a testcase can be added.
#    See custom tests below.
#
# The 'check' target runs the testcases including SWIG invocation,
# C/C++ compilation, target language compilation (if any) and runtime
# test (if there is an associated 'runme' test).
# The 'partialcheck' target only invokes SWIG.
# The 'all' target is the same as the 'check' target but also includes
# known broken testcases.
# The 'clean' target cleans up.
#
# Note that the RUNTOOL, COMPILETOOL and SWIGTOOL variables can be used
# for invoking tools for the runtime tests and target language 
# compiler (eg javac), and on SWIG respectively. For example, valgrind
# can be used for memory checking of the runtime tests using:
#   make RUNTOOL="valgrind --leak-check=full"
# and valgrind can be used when invoking SWIG using:
#   make SWIGTOOL="valgrind --tool=memcheck"
#
# An individual test run can be debugged easily:
#   make director_string.cpptest RUNTOOL="gdb --args"
#
# The variables below can be overridden after including this makefile
#######################################################################

#######################################################################
# Variables
#######################################################################

ifneq (,$(USE_VALGRIND))
VALGRIND_OPT = --leak-check=full
RUNTOOL    = valgrind $(VALGRIND_OPT)
else
RUNTOOL    =
endif
COMPILETOOL=
SWIGTOOL   =

SWIGEXE   = $(top_builddir)/swig
SWIG_LIB_DIR = $(top_srcdir)/Lib
TEST_SUITE = test-suite
EXAMPLES   = Examples
CXXSRCS    = 
CSRCS      = 
TARGETPREFIX = 
TARGETSUFFIX = 
SWIGOPT    = -outcurrentdir -I$(top_srcdir)/$(EXAMPLES)/$(TEST_SUITE)
INCLUDES   = -I$(top_srcdir)/$(EXAMPLES)/$(TEST_SUITE)
LIBS       = -L.
LIBPREFIX  = lib
ACTION     = check
INTERFACEDIR = ../
SRCDIR     = $(srcdir)/
SCRIPTDIR  = $(srcdir)

# Regenerate Makefile if Makefile.in or config.status have changed.
Makefile: $(srcdir)/Makefile.in ../../../config.status
	cd ../../../ && $(SHELL) ./config.status $(EXAMPLES)/$(TEST_SUITE)/$(LANGUAGE)/Makefile

#
# Please keep test cases in alphabetical order.
# Note that any whitespace after the last entry in each list will break make
#

# Broken C++ test cases. (Can be run individually using: make testcase.cpptest)
CPP_TEST_BROKEN += \
	constants \
	cpp_broken \
	director_nested_class \
	exception_partial_info \
	extend_variable \
	li_boost_shared_ptr_template \
	nested_private \
	overload_complicated \
	rename_camel \
	template_default_pointer \
	template_private_assignment \
	template_expr \
	$(CPP11_TEST_BROKEN)


# Broken C test cases. (Can be run individually using: make testcase.ctest)
C_TEST_BROKEN += \
	tag_no_clash_with_variable

# C++ test cases. (Can be run individually using: make testcase.cpptest)
CPP_TEST_CASES += \
	abstract_access \
	abstract_inherit \
	abstract_inherit_ok \
	abstract_signature \
	abstract_typedef \
	abstract_typedef2 \
	abstract_virtual \
	access_change \
	add_link \
	aggregate \
	allowexcept \
	allprotected \
	allprotected_not \
	anonymous_bitfield \
	apply_signed_char \
	apply_strings \
	argout \
	array_member \
	array_typedef_memberin \
	arrayref \
	arrays_dimensionless \
	arrays_global \
	arrays_global_twodim \
	arrays_scope \
	autodoc \
	bloody_hell \
	bools \
	catches \
	cast_operator \
	casts \
	char_binary \
	char_strings \
	chartest \
	class_forward \
	class_ignore \
	class_scope_weird \
	compactdefaultargs \
	const_const_2 \
	constant_directive \
	constant_pointers \
	constover \
	constructor_copy \
	constructor_exception \
	constructor_explicit \
	constructor_ignore \
	constructor_rename \
	constructor_value \
	contract \
	conversion \
	conversion_namespace \
	conversion_ns_template \
	conversion_operators \
	cplusplus_throw \
	cpp_basic \
	cpp_enum \
	cpp_namespace \
	cpp_nodefault \
	cpp_static \
	cpp_typedef \
	curiously_recurring_template_pattern \
	default_args \
	default_arg_values \
	default_constructor \
	defvalue_constructor \
	derived_byvalue \
	derived_nested \
	destructor_reprotected \
	director_abstract \
	director_alternating \
	director_basic \
	director_binary_string \
	director_classes \
	director_classic \
	director_constructor \
	director_default \
	director_detect \
	director_enum \
	director_exception \
	director_extend \
	director_finalizer \
	director_frob \
	director_ignore \
	director_keywords \
	director_namespace_clash \
	director_nested \
	director_nspace \
	director_nspace_director_name_collision \
	director_overload \
	director_overload2 \
	director_pass_by_value \
	director_primitives \
	director_property \
	director_protected \
	director_protected_overloaded \
	director_redefined \
	director_ref \
	director_smartptr \
	director_thread \
	director_unroll \
	director_using \
	director_void \
	director_wombat \
	disown \
	dynamic_cast \
	empty \
	enum_ignore \
	enum_plus \
	enum_rename \
	enum_scope_template \
	enum_template \
	enum_thorough \
	enum_var \
	equality \
	evil_diamond \
	evil_diamond_ns \
	evil_diamond_prop \
	exception_classname \
	exception_order \
	extend \
	extend_constructor_destructor \
	extend_default \
	extend_placement \
	extend_special_variables \
	extend_template \
	extend_template_method \
	extend_template_ns \
	extend_typedef_class \
	extern_c \
	extern_namespace \
	extern_throws \
	expressions \
	features \
	fragments \
	friends \
	friends_template \
	funcptr_cpp \
	fvirtual \
	global_namespace \
	global_ns_arg \
	global_scope_types \
	global_vars \
	grouping \
	ignore_parameter \
	import_nomodule \
	inherit \
	inherit_member \
	inherit_missing \
	inherit_same_name \
	inherit_target_language \
	inherit_void_arg \
	inline_initializer \
	insert_directive \
	keyword_rename \
	kind \
	kwargs_feature \
	langobj \
	li_attribute \
	li_attribute_template \
	li_boost_shared_ptr \
	li_boost_shared_ptr_bits \
	li_boost_shared_ptr_template \
	li_boost_shared_ptr_attribute \
	li_carrays_cpp \
	li_cdata_cpp \
	li_cpointer_cpp \
	li_std_auto_ptr \
	li_stdint \
	li_swigtype_inout \
	li_typemaps \
	li_typemaps_apply \
	li_windows \
	long_long_apply \
	memberin_extend \
	member_funcptr_galore \
	member_pointer \
	member_template \
	minherit \
	minherit2 \
	mixed_types \
	multiple_inheritance \
	multiple_inheritance_abstract \
	multiple_inheritance_interfaces \
	multiple_inheritance_nspace \
	multiple_inheritance_shared_ptr \
	name_cxx \
	name_warnings \
	namespace_class \
	namespace_enum \
	namespace_extend \
	namespace_forward_declaration \
	namespace_nested \
	namespace_spaces \
	namespace_template \
	namespace_typedef_class \
	namespace_typemap \
	namespace_union \
	namespace_virtual_method \
	nspace \
	nspace_extend \
	naturalvar \
	naturalvar_more \
	naturalvar_onoff \
	nested_class \
	nested_directors \
	nested_comment \
	nested_ignore \
	nested_scope \
	nested_template_base \
	nested_workaround \
	newobject1 \
	null_pointer \
	operator_overload \
	operator_overload_break \
	operator_pointer_ref \
	operbool \
	ordering \
	overload_arrays \
	overload_bool \
	overload_copy \
	overload_extend \
	overload_method \
	overload_numeric \
	overload_polymorphic \
	overload_rename \
	overload_return_type \
	overload_simple \
	overload_subtype \
	overload_template \
	overload_template_fast \
	pointer_reference \
	preproc_constants \
	primitive_ref \
	private_assign \
	proxycode \
	protected_rename \
	pure_virtual \
	redefined \
	redefined_not \
	refcount \
	reference_global_vars \
	register_par \
	rename1 \
	rename2 \
	rename3 \
	rename4 \
	rename_rstrip_encoder \
	rename_scope \
	rename_simple \
	rename_strip_encoder \
	rename_pcre_encoder \
	rename_pcre_enum \
	rename_predicates \
	rename_wildcard \
	restrict_cplusplus \
	return_const_value \
	return_value_scope \
	rname \
	samename \
	sizet \
	smart_pointer_const \
	smart_pointer_const2 \
	smart_pointer_const_overload \
	smart_pointer_extend \
	smart_pointer_ignore \
	smart_pointer_member \
	smart_pointer_multi \
	smart_pointer_multi_typedef \
	smart_pointer_namespace \
	smart_pointer_namespace2 \
	smart_pointer_not \
	smart_pointer_overload \
	smart_pointer_protected \
	smart_pointer_rename \
	smart_pointer_simple \
	smart_pointer_static \
	smart_pointer_template_const_overload \
	smart_pointer_template_defaults_overload \
	smart_pointer_templatemethods \
	smart_pointer_templatevariables \
	smart_pointer_typedef \
	special_variables \
	special_variable_attributes \
	special_variable_macros \
	static_array_member \
	static_const_member \
	static_const_member_2 \
	string_constants \
	struct_initialization_cpp \
	struct_value \
	swig_exception \
	symbol_clash \
	template_arg_replace \
	template_arg_scope \
	template_arg_typename \
	template_array_numeric \
	template_basic \
	template_base_template \
	template_classes \
	template_const_ref \
	template_construct \
	template_templated_constructors \
	template_default \
	template_default2 \
	template_default_arg \
	template_default_arg_overloaded \
	template_default_arg_overloaded_extend \
	template_default_arg_virtual_destructor \
	template_default_cache \
	template_default_class_parms \
	template_default_class_parms_typedef \
	template_default_inherit \
	template_default_qualify \
	template_default_vw \
	template_enum \
	template_enum_ns_inherit \
	template_enum_typedef \
	template_explicit \
	template_extend1 \
	template_extend2 \
	template_extend_overload \
	template_extend_overload_2 \
	template_forward \
	template_inherit \
	template_inherit_abstract \
	template_int_const \
	template_keyword_in_type \
	template_methods \
	template_namespace_forward_declaration \
	template_using_directive_and_declaration_forward \
	template_nested \
	template_nested_typemaps \
	template_ns \
	template_ns2 \
	template_ns3 \
	template_ns4 \
	template_ns_enum \
	template_ns_enum2 \
	template_ns_inherit \
	template_ns_scope \
	template_partial_arg \
	template_partial_specialization \
	template_partial_specialization_typedef \
	template_qualifier \
	template_ref_type \
	template_rename \
	template_retvalue \
	template_specialization \
	template_specialization_defarg \
	template_specialization_enum \
	template_static \
	template_tbase_template \
	template_template_parameters \
	template_typedef \
	template_typedef_class_template \
	template_typedef_cplx \
	template_typedef_cplx2 \
	template_typedef_cplx3 \
	template_typedef_cplx4 \
	template_typedef_cplx5 \
	template_typedef_funcptr \
	template_typedef_inherit \
	template_typedef_ns \
	template_typedef_ptr \
	template_typedef_rec \
	template_typedef_typedef \
	template_typemaps \
	template_typemaps_typedef \
	template_typemaps_typedef2 \
	template_using \
	template_virtual \
	template_whitespace \
	threads \
	threads_exception \
	throw_exception \
	typedef_array_member \
	typedef_class \
	typedef_funcptr \
	typedef_inherit \
	typedef_mptr \
	typedef_reference \
	typedef_scope \
	typedef_sizet \
	typedef_struct_cpp \
	typedef_typedef \
	typemap_arrays \
	typemap_array_qualifiers \
	typemap_delete \
	typemap_directorout \
	typemap_documentation \
	typemap_global_scope \
	typemap_manyargs \
	typemap_namespace \
	typemap_ns_using \
	typemap_numinputs \
	typemap_template \
	typemap_template_parm_typedef \
	typemap_out_optimal \
	typemap_qualifier_strip \
	typemap_variables \
	typemap_various \
	typename \
	types_directive \
	unicode_strings \
	union_scope \
	using1 \
	using2 \
	using_composition \
	using_directive_and_declaration \
	using_directive_and_declaration_forward \
	using_extend \
	using_inherit \
	using_namespace \
	using_namespace_loop \
	using_pointers \
	using_private \
	using_protected \
	valuewrapper \
	valuewrapper_base \
	valuewrapper_const \
	valuewrapper_opaque \
	varargs \
	varargs_overload \
	variable_replacement \
	virtual_destructor \
	virtual_poly \
	virtual_vs_nonvirtual_base \
	voidtest \
	wallkw \
	wrapmacro

# C++11 test cases.
CPP11_TEST_CASES = \
	cpp11_alignment \
	cpp11_alternate_function_syntax \
	cpp11_constexpr \
	cpp11_decltype \
	cpp11_default_delete \
	cpp11_delegating_constructors \
	cpp11_director_enums \
	cpp11_explicit_conversion_operators \
	cpp11_final_override \
	cpp11_function_objects \
	cpp11_inheriting_constructors \
	cpp11_initializer_list \
	cpp11_initializer_list_extend \
	cpp11_lambda_functions \
	cpp11_li_std_array \
	cpp11_noexcept \
	cpp11_null_pointer_constant \
	cpp11_raw_string_literals \
	cpp11_result_of \
	cpp11_rvalue_reference \
	cpp11_rvalue_reference2 \
	cpp11_rvalue_reference3 \
	cpp11_sizeof_object \
	cpp11_static_assert \
	cpp11_strongly_typed_enumerations \
	cpp11_thread_local \
	cpp11_template_double_brackets \
	cpp11_template_explicit \
	cpp11_template_typedefs \
	cpp11_type_traits \
	cpp11_type_aliasing \
	cpp11_uniform_initialization \
	cpp11_unrestricted_unions \
	cpp11_userdefined_literals \

# Broken C++11 test cases.
CPP11_TEST_BROKEN = \
#	cpp11_hash_tables \           # not fully implemented yet
#	cpp11_variadic_templates \    # Broken for some languages (such as Java)
#	cpp11_reference_wrapper \     # No typemaps


#
# Put all the heavy STD/STL cases here, where they can be skipped if needed
#
CPP_STD_TEST_CASES += \
	director_string \
	ignore_template_constructor \
	li_std_combinations \
	li_std_deque \
	li_std_except \
	li_std_except_as_class \
	li_std_map \
	li_std_pair \
	li_std_pair_using \
	li_std_string \
	li_std_vector \
	li_std_vector_enum \
	li_std_vector_member_var\
	li_std_vector_ptr \
	smart_pointer_inherit \
	template_typedef_fnc \
	template_type_namespace \
	template_opaque
#        li_std_list


ifndef SKIP_CPP_STD_CASES
CPP_TEST_CASES += ${CPP_STD_TEST_CASES}
endif

ifneq (,$(HAVE_CXX11_COMPILER))
CPP_TEST_CASES += $(CPP11_TEST_CASES)
endif

# C test cases. (Can be run individually using: make testcase.ctest)
C_TEST_CASES += \
	arrays \
	bom_utf8 \
	c_delete \
	c_delete_function \
	char_constant \
	const_const \
	constant_expr \
	empty_c \
	enums \
	enum_forward \
	enum_macro \
	enum_missing \
	extern_declaration \
	funcptr \
	function_typedef \
	global_functions \
	immutable_values \
	inctest \
	infinity \
	integers \
	keyword_rename_c \
	lextype \
	li_carrays \
	li_cdata \
	li_cmalloc \
	li_constraints \
	li_cpointer \
	li_math \
	long_long \
	memberin_extend_c \
	name \
	nested \
	nested_extend_c \
	nested_structs \
	newobject2 \
	overload_extend_c \
	overload_extend2 \
	preproc \
	preproc_constants_c \
	preproc_defined \
	preproc_include \
	preproc_line_file \
	ret_by_value \
	simple_array \
	sizeof_pointer \
	sneaky1 \
	string_simple \
	struct_rename \
	struct_initialization \
	typedef_struct \
	typemap_subst \
	union_parameter \
	unions


# Multi-module C++ test cases . (Can be run individually using make testcase.multicpptest)
MULTI_CPP_TEST_CASES += \
	clientdata_prop \
	imports \
	import_stl \
	packageoption \
	mod \
	template_typedef_import \
	multi_import

# Custom tests - tests with additional commandline options
wallkw.cpptest: SWIGOPT += -Wallkw
preproc_include.ctest: SWIGOPT += -includeall

# Allow modules to define temporarily failing tests.
C_TEST_CASES := $(filter-out $(FAILING_C_TESTS),$(C_TEST_CASES))
CPP_TEST_CASES := $(filter-out $(FAILING_CPP_TESTS),$(CPP_TEST_CASES))
MULTI_CPP_TEST_CASES := $(filter-out $(FAILING_MULTI_CPP_TESTS),$(MULTI_CPP_TEST_CASES))


NOT_BROKEN_TEST_CASES =	$(CPP_TEST_CASES:=.cpptest) \
			$(C_TEST_CASES:=.ctest) \
			$(MULTI_CPP_TEST_CASES:=.multicpptest) \
			$(EXTRA_TEST_CASES)

BROKEN_TEST_CASES = 	$(CPP_TEST_BROKEN:=.cpptest) \
			$(C_TEST_BROKEN:=.ctest)

ALL_CLEAN = 		$(CPP_TEST_CASES:=.clean) \
			$(C_TEST_CASES:=.clean) \
			$(MULTI_CPP_TEST_CASES:=.clean) \
			$(CPP_TEST_BROKEN:=.clean) \
			$(C_TEST_BROKEN:=.clean)

#######################################################################
# Error test suite has its own set of test cases
#######################################################################
ifneq (,$(ERROR_TEST_CASES))
check: $(ERROR_TEST_CASES)
else

#######################################################################
# The following applies for all module languages
#######################################################################
all: $(NOT_BROKEN_TEST_CASES) $(BROKEN_TEST_CASES)

broken: $(BROKEN_TEST_CASES)

check: $(NOT_BROKEN_TEST_CASES)
	@echo $(words $^) $(LANGUAGE) tests passed

check-c: $(C_TEST_CASES:=.ctest)

check-cpp: $(CPP_TEST_CASES:=.cpptest)

check-cpp11: $(CPP11_TEST_CASES:=.cpptest)

check-failing-test = \
	$(MAKE) -s $1.$2 >/dev/null 2>/dev/null && echo "Failing test $1 passed."

check-failing:
	+-$(foreach t,$(FAILING_C_TESTS),$(call check-failing-test,$t,ctest);)
	+-$(foreach t,$(FAILING_CPP_TESTS),$(call check-failing-test,$t,cpptest);)
	+-$(foreach t,$(FAILING_MULTI_CPP_TESTS),$(call check-failing-test,$t,multicpptest);)
endif

# partialcheck target runs SWIG only, ie no compilation or running of tests (for a subset of languages)
partialcheck:
	$(MAKE) check CC=true CXX=true LDSHARED=true CXXSHARED=true RUNTOOL=true COMPILETOOL=true

swig_and_compile_cpp =  \
	$(MAKE) -f $(top_builddir)/$(EXAMPLES)/Makefile SRCDIR='$(SRCDIR)' CXXSRCS='$(CXXSRCS)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	INCLUDES='$(INCLUDES)' SWIGOPT='$(SWIGOPT)' NOLINK=true \
	TARGET='$(TARGETPREFIX)$*$(TARGETSUFFIX)' INTERFACEDIR='$(INTERFACEDIR)' INTERFACE='$*.i' \
	$(LANGUAGE)$(VARIANT)_cpp

swig_and_compile_c =  \
	$(MAKE) -f $(top_builddir)/$(EXAMPLES)/Makefile SRCDIR='$(SRCDIR)' CSRCS='$(CSRCS)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	INCLUDES='$(INCLUDES)' SWIGOPT='$(SWIGOPT)' NOLINK=true \
	TARGET='$(TARGETPREFIX)$*$(TARGETSUFFIX)' INTERFACEDIR='$(INTERFACEDIR)' INTERFACE='$*.i' \
	$(LANGUAGE)$(VARIANT)

swig_and_compile_multi_cpp = \
	for f in `cat $(top_srcdir)/$(EXAMPLES)/$(TEST_SUITE)/$*.list` ; do \
	  $(MAKE) -f $(top_builddir)/$(EXAMPLES)/Makefile SRCDIR='$(SRCDIR)' CXXSRCS='$(CXXSRCS)' \
	  SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	  LIBS='$(LIBS)' INCLUDES='$(INCLUDES)' SWIGOPT='$(SWIGOPT)' NOLINK=true \
	  TARGET="$(TARGETPREFIX)$${f}$(TARGETSUFFIX)" INTERFACEDIR='$(INTERFACEDIR)' INTERFACE="$$f.i" \
	  $(LANGUAGE)$(VARIANT)_cpp; \
	done

swig_and_compile_external =  \
	$(MAKE) -f $(top_builddir)/$(EXAMPLES)/Makefile SRCDIR='$(SRCDIR)' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	TARGET='$*_wrap_hdr.h' \
	$(LANGUAGE)$(VARIANT)_externalhdr; \
	$(MAKE) -f $(top_builddir)/$(EXAMPLES)/Makefile SRCDIR='$(SRCDIR)' CXXSRCS='$(CXXSRCS) $*_external.cxx' \
	SWIG_LIB_DIR='$(SWIG_LIB_DIR)' SWIGEXE='$(SWIGEXE)' \
	INCLUDES='$(INCLUDES)' SWIGOPT='$(SWIGOPT)' NOLINK=true \
	TARGET='$(TARGETPREFIX)$*$(TARGETSUFFIX)' INTERFACEDIR='$(INTERFACEDIR)' INTERFACE='$*.i' \
	$(LANGUAGE)$(VARIANT)_cpp

swig_and_compile_runtime = \

setup = \
	if [ -f $(SCRIPTDIR)/$(SCRIPTPREFIX)$*$(SCRIPTSUFFIX) ]; then	  \
	  echo "$(ACTION)ing $(LANGUAGE) testcase $* (with run test)" ; \
	else								  \
	  echo "$(ACTION)ing $(LANGUAGE) testcase $*" ;		  \
	fi;



#######################################################################
# Clean
#######################################################################
clean: $(ALL_CLEAN)

distclean: clean
	@rm -f Makefile

.PHONY: all check partialcheck broken clean distclean 

