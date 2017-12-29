;; The SWIG modules have "passive" Linkage, i.e., they don't generate
;; Guile modules (namespaces) but simply put all the bindings into the
;; current module.  That's enough for such a simple test.
(dynamic-call "scm_init_pointer_in_out_module" (dynamic-link "./libpointer_in_out"))
(load "../schemerunme/pointer_in_out.scm")
