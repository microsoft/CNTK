;; The SWIG modules have "passive" Linkage, i.e., they don't generate
;; Guile modules (namespaces) but simply put all the bindings into the
;; current module.  That's enough for such a simple test.
(dynamic-call "scm_init_throw_exception_module" (dynamic-link "./libthrow_exception"))

(define-macro (check-throw form)
  `(catch 'swig-exception
     (lambda ()
       ,form
       (error "Check failed (returned normally): " ',form))
     (lambda (key result)
       result)))

(define-macro (check-throw-error form)
  `(let ((result (check-throw ,form)))
     (test-is-Error result)))

(let ((foo (new-Foo)))
  (let ((result (check-throw (Foo-test-int foo))))
    (if (not (eqv? result 37))
	(error "Foo-test-int failed, returned " result)))
  (let ((result (check-throw (Foo-test-multi foo 1))))
    (if (not (eqv? result 37))
	(error "Foo-test-multi 1 failed, returned " result)))
  (let ((result (check-throw (Foo-test-msg foo))))
    (if (not (and (string? result)
		  (string=? result "Dead")))
	(error "Foo-test-msg failed, returned " result)))
  (let ((result (check-throw (Foo-test-multi foo 2))))
    (if (not (and (string? result)
		  (string=? result "Dead")))
	(error "Foo-test-multi 2 failed, returned " result)))
  (check-throw-error (Foo-test-cls foo))
  (check-throw-error (Foo-test-multi foo 3))
  (check-throw-error (Foo-test-cls-ptr foo))
  (check-throw-error (Foo-test-cls-ref foo))
  ;; Namespace stuff
  (let ((result (check-throw (Foo-test-enum foo))))
    (if (not (eqv? result (enum2)))
	(error "Foo-test-enum failed, returned " result)))
  (check-throw-error (Foo-test-cls-td foo))
  (check-throw-error (Foo-test-cls-ptr-td foo))
  (check-throw-error (Foo-test-cls-ref-td foo)))
  			      
(exit 0)
