(load "newobject2.so")

(define f (new-Foo))

(Foo-dummy-set f 14)
(if (not (= (Foo-dummy-get f) 14))
  (error "Bad dummy value"))

(if (not (= (fooCount) 0))
  (error "Bad foo count 1"))

(define f2 (makeFoo))

(if (not (= (fooCount) 1))
  (error "Bad foo count 2"))

(Foo-dummy-set f2 16)
(if (not (= (Foo-dummy-get f2) 16))
  (error "Bad dummy value for f2"))

(set! f #f)
(set! f2 #f)

(gc #t)

(if (not (= (fooCount) -1))
  (error "Bad foo count 3"))

(exit 0)
