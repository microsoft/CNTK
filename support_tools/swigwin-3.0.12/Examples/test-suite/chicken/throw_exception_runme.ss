(load "throw_exception.so")

(define-macro (check-throw expr check)
 `(if (handle-exceptions exvar (if ,check #f (begin (print "Error executing: " ',expr " " exvar) (exit 1))) ,expr #t)
    (print "Expression did not throw an error: " ',expr)))

(define f (new-Foo))

(check-throw (Foo-test-int f) (= exvar 37))
(check-throw (Foo-test-msg f) (string=? exvar "Dead"))
(check-throw (Foo-test-cls f) (test-is-Error exvar))
(check-throw (Foo-test-cls-ptr f) (test-is-Error exvar))
(check-throw (Foo-test-cls-ref f) (test-is-Error exvar))
(check-throw (Foo-test-cls-td f) (test-is-Error exvar))
(check-throw (Foo-test-cls-ptr-td f) (test-is-Error exvar))
(check-throw (Foo-test-cls-ref-td f) (test-is-Error exvar))
(check-throw (Foo-test-enum f) (= exvar (enum2)))

; don't know how to test this... it is returning a SWIG wrapped int *
;(check-throw (Foo-test-array f) (equal? exvar '(0 1 2 3 4 5 6 7 8 9)))

(check-throw (Foo-test-multi f 1) (= exvar 37))
(check-throw (Foo-test-multi f 2) (string=? exvar "Dead"))
(check-throw (Foo-test-multi f 3) (test-is-Error exvar))

(set! f #f)
(gc #t)

(exit 0)
