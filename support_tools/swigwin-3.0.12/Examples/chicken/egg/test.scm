(require-extension single)
(require-extension multi)

(define f (make <Foo>))
(slot-set! f 'a 3)
(print (slot-ref f 'a))

(define b (make <Bar>))
(slot-set! b 'b 2)
(print (slot-ref b 'b))

(define b2 (make <Bar2>))
(slot-set! b2 'b 4)
(slot-set! b2 'c 6)
(print (slot-ref b2 'b))
(print (slot-ref b2 'c))

(exit 0)
