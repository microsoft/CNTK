;;; This is the union runtime testcase. It ensures that values within a
;;; union embedded within a struct can be set and read correctly.

;; The SWIG modules have "passive" Linkage, i.e., they don't generate
;; Guile modules (namespaces) but simply put all the bindings into the
;; current module.  That's enough for such a simple test.
(dynamic-call "scm_init_unions_module" (dynamic-link "./libunions"))
(load "../schemerunme/unions.scm")
