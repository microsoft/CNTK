(define-macro (check test)
  `(if (not ,test) (error "Error in test" ',test)))

(define b (make <Bar>))
(check (= (bar b) 1))

(define f (make <Foo>))
(check (= (foo f) 2))

(define fb (make <FooBar>))
(check (= (bar fb) 1))
(check (= (foo fb) 2))
(check (= (fooBar fb) 3))

(define id1 (make <IgnoreDerived1>))
(check (= (bar id1) 1))
(check (= (ignorederived1 id1) 7))

(define id2 (make <IgnoreDerived2>))
(check (= (bar id2) 1))
(check (= (ignorederived2 id2) 8))

(define id3 (make <IgnoreDerived3>))
(check (= (bar id3) 1))
(check (= (ignorederived3 id3) 9))

(define id4 (make <IgnoreDerived4>))
(check (= (bar id4) 1))
(check (= (ignorederived4 id4) 10))

(exit 0)
