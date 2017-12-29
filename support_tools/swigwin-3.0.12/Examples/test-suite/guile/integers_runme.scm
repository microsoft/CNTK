;; The SWIG modules have "passive" Linkage, i.e., they don't generate
;; Guile modules (namespaces) but simply put all the bindings into the
;; current module.  That's enough for such a simple test.
(dynamic-call "scm_init_integers_module" (dynamic-link "./libintegers"))

(define-macro (throws-exception? form)
  `(catch #t
     (lambda () ,form #f)
     (lambda args #t)))

(load "../schemerunme/integers.scm")
