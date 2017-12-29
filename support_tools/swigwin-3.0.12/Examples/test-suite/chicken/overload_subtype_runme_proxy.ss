(load "./overload_subtype.so")

(define f (make <Foo>))
(define b (make <Bar>))

(if (not (= (spam f) 1))
  (error "Error in foo"))

(if (not (= (spam b) 2))
  (error "Error in bar"))

(exit 0)
