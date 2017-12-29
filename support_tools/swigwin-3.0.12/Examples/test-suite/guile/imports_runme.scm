;;; This file is part of a test for SF bug #231619. 
;;; It shows that the %import directive does not work properly in SWIG
;;; 1.3a5:  Type information is not properly generated if a base class
;;; comes from an %import-ed file. 

;; The SWIG modules have "passive" Linkage, i.e., they don't generate
;; Guile modules (namespaces) but simply put all the bindings into the
;; current module.  That's enough for such a simple test.
(dynamic-call "scm_init_imports_a_module" (dynamic-link "./libimports_a"))
(dynamic-call "scm_init_imports_b_module" (dynamic-link "./libimports_b"))
(load "../schemerunme/imports.scm")
