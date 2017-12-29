(load "newobject2.so")

(define f (make <Foo>))

(slot-set! f 'dummy 14)
(if (not (= (slot-ref f 'dummy) 14))
  (error "Bad dummy value"))

(if (not (= (fooCount) 0))
  (error "Bad foo count 1"))

(define f2 (makeFoo))

(if (not (= (fooCount) 1))
  (error "Bad foo count 2"))

(slot-set! f2 'dummy 16)
(if (not (= (slot-ref f2 'dummy) 16))
  (error "Bad dummy value for f2"))

(set! f #f)
(set! f2 #f)

(gc #t)

(if (not (= (fooCount) -1))
  (error "Bad foo count 3"))

(exit 0)
