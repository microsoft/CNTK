(require 'cpp_basic)

(define-macro (check test)
  `(if (not ,test) (error "Error in test " ',test)))

(define f (make <Foo> 4))
(check (= (slot-ref f 'num) 4))
(slot-set! f 'num -17)
(check (= (slot-ref f 'num) -17))

(define b (make <Bar>))

(slot-set! b 'fptr f)
(check (= (slot-ref (slot-ref b 'fptr) 'num) -17))
(check (= (test b -3 (slot-ref b 'fptr)) -5))
(slot-set! f 'num 12)
(check (= (slot-ref (slot-ref b 'fptr) 'num) 12))

(check (= (slot-ref (slot-ref b 'fref) 'num) -4))
(check (= (test b 12 (slot-ref b 'fref)) 23))
;; references don't take ownership, so if we didn't define this here it might get garbage collected
(define f2 (make <Foo> 23))
(slot-set! b 'fref f2)
(check (= (slot-ref (slot-ref b 'fref) 'num) 23))
(check (= (test b -3 (slot-ref b 'fref)) 35))

(check (= (slot-ref (slot-ref b 'fval) 'num) 15))
(check (= (test b 3 (slot-ref b 'fval)) 33))
(slot-set! b 'fval (make <Foo> -15))
(check (= (slot-ref (slot-ref b 'fval) 'num) -15))
(check (= (test b 3 (slot-ref b 'fval)) -27))

(define f3 (testFoo b 12 (slot-ref b 'fref)))
(check (= (slot-ref f3 'num) 32))

;; now test global
(define f4 (make <Foo> 6))
(Bar-global-fptr f4)
(check (= (slot-ref (Bar-global-fptr) 'num) 6))
(slot-set! f4 'num 8)
(check (= (slot-ref (Bar-global-fptr) 'num) 8))

(check (= (slot-ref (Bar-global-fref) 'num) 23))
(Bar-global-fref (make <Foo> -7))
(check (= (slot-ref (Bar-global-fref) 'num) -7))

(check (= (slot-ref (Bar-global-fval) 'num) 3))
(Bar-global-fval (make <Foo> -34))
(check (= (slot-ref (Bar-global-fval) 'num) -34))

;; Now test function pointers
(define func1ptr (get-func1-ptr))
(define func2ptr (get-func2-ptr))

(slot-set! f 'num 4)
(check (= (func1 f 2) 16))
(check (= (func2 f 2) -8))

(slot-set! f 'func-ptr func1ptr)
(check (= (test-func-ptr f 2) 16))
(slot-set! f 'func-ptr func2ptr)
(check (= (test-func-ptr f 2) -8))

(exit 0)
