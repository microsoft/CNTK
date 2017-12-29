;; Little helper functions and macros for the run tests

(use-modules (ice-9 format))

(define (test-error error-format . args)
  (display "Runtime check failed. ")
  (apply format #t error-format args)
  (newline)
  (exit 1))

(define-macro (expect-true form)
  `(if (not ,form)
       (test-error "Expected true value of ~A" ',form)))

(define-macro (expect-false form)
  `(if ,form
       (test-error "Expected false value of ~A" ',form)))

(define-macro (expect-result expected-result-form equal? form)
  `(let ((expected-result ,expected-result-form)
	 (result ,form))
     (if (not (,equal? result expected-result))
	 (test-error "The result of ~A was ~A, expected ~A, which is not ~A"
		     ',form result expected-result ',equal?))))

(define-macro (expect-throw tag-form form)
  `(let ((tag ,tag-form))
     (if (catch #t
		(lambda ()
		  ,form
		  #t)
		(lambda (key . args)
		  (if (eq? key ,tag-form)
		      #f
		      (test-error "The form ~A threw to ~A (expected a throw to ~A)"
				  ',form key tag))))
	 (test-error "The form ~A returned normally (expected a throw to ~A)"))))
