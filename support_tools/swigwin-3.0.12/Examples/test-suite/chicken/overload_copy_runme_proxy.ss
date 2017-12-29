(load "./overload_copy.so")

(define f (make <Foo>))
(define g (make <Foo> f))

(exit 0)
