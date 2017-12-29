;; The SWIG modules have "passive" Linkage, i.e., they don't generate
;; Guile modules (namespaces) but simply put all the bindings into the
;; current module.  That's enough for such a simple test.
(dynamic-call "scm_init_overload_complicated_module" (dynamic-link "./liboverload_complicated"))

(define-macro (check form)
  `(if (not ,form)
       (error "Check failed: " ',form)))

(define (=~ a b)
  (< (abs (- a b)) 1e-8))

;; Check first method
(check (=~ (foo 1 2 "bar" 4) 15))

;; Check second method
(check (=~ (foo 1 2) 4811.4))
(check (=~ (foo 1 2 3.2) 4797.2))
(check (=~ (foo 1 2 3.2 #\Q) 4798.2))

(exit 0)
