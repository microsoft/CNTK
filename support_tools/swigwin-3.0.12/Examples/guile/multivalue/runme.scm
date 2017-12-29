;;;; Show the three different ways to deal with multiple return values

(load-extension "./libexample" "scm_init_example_module")

;;; Multiple values as lists. By default, if more than one value is to
;;; be returned, a list of the values is created and returned.  The
;;; procedure divide-l does so:

(let* ((quotient/remainder (divide-l 37 5))
       ;; divide-l returns a list of the two values, so get them:
       (quotient (car quotient/remainder))
       (remainder (cadr quotient/remainder)))
  (display "37 divided by 5 is ")
  (display quotient)
  (display ", remainder ")
  (display remainder)
  (newline))

;;; Multiple values as vectors.  You can get vectors instead of lists
;;; if you want:

(let* ((quotient-remainder-vector (divide-v 40 7))
       ;; divide-v returns a vector of two values, so get them:
       (quotient (vector-ref quotient-remainder-vector 0))
       (remainder (vector-ref quotient-remainder-vector 1)))
  (display "40 divided by 7 is ")
  (display quotient)
  (display ", remainder ")
  (display remainder)
  (newline))

;;; Multiple values for multiple-value continuations. (The most
;;; elegant way.)  You can get multiple values passed to the
;;; multiple-value continuation, as created by `call-with-values'.

(call-with-values (lambda ()
		    ;; the "producer" procedure
		    (divide-mv 91 13))
		  (lambda (quotient remainder)
		    ;; the "consumer" procedure
		    (display "91 divided by 13 is ")
		    (display quotient)
		    (display ", remainder ")
		    (display remainder)
		    (newline)))

;;; SRFI-8 has a very convenient macro for this construction:

(use-modules (srfi srfi-8))

;;; If your Guile is too old, you can define the receive macro yourself:
;;;
;;; (define-macro (receive vars vals . body)
;;;  `(call-with-values (lambda () ,vals)
;;;     (lambda ,vars ,@body)))

(receive (quotient remainder)
    (divide-mv 111 19) ; the "producer" form
  ;; In the body, `quotient' and `remainder' are bound to the two
  ;; values.
  (display "111 divided by 19 is ")
  (display quotient)
  (display ", remainder ")
  (display remainder)
  (newline))

(exit 0)
