(define-macro (check test)
  `(if (not ,test) (error "Error in test" ',test)))

(b "hello")
(check (string=? (b) "hello"))

(define sa (make <A>))
(slot-set! sa 'x 5)
(a sa)
(check (= (slot-ref (a) 'x) 5))

(ap sa)
(check (= (slot-ref (ap) 'x) 5))
(slot-set! sa 'x 10)
(check (= (slot-ref (ap) 'x) 10))

(define sa2 (make <A>))
(slot-set! sa2 'x -4)
(cap sa2)
(check (= (slot-ref (cap) 'x) -4))
(slot-set! sa2 'x -7)
(check (= (slot-ref (cap) 'x) -7))

(check (= (slot-ref (ar) 'x) 5))
(ar sa2)
(check (= (slot-ref (ar) 'x) -7))

(x 4)
(check (= (x) 4))

(exit 0)
