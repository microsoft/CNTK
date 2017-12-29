;;; This is the union runtime testcase. It ensures that values within a
;;; union embedded within a struct can be set and read correctly.

;; The SWIG modules have "passive" Linkage, i.e., they don't generate
;; Guile modules (namespaces) but simply put all the bindings into the
;; current module.  That's enough for such a simple test.
(dynamic-call "scm_init_li_typemaps_module" (dynamic-link "./libli_typemaps"))
(load "../schemerunme/li_typemaps.scm")

(let ((lst (inoutr-int2 3 -2)))
  (if (not (and (= (car lst) 3) (= (cadr lst) -2)))
    (error "Error in inoutr-int2")))

(let ((lst (out-foo 4)))
  (if (not (and (= (Foo-a-get (car lst)) 4) (= (cadr lst) 8)))
    (error "Error in out-foo")))

(exit 0)
