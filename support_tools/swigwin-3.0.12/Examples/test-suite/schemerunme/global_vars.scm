(define-macro (check test)
  `(if (not ,test) (error "Error in test" ',test)))

(b "hello")
(check (string=? (b) "hello"))

(define sa (new-A))
(A-x-set sa 5)
(a sa)
(check (= (A-x-get (a)) 5))

(ap sa)
(check (= (A-x-get (ap)) 5))
(A-x-set sa 10)
(check (= (A-x-get (ap)) 10))

(define sa2 (new-A))
(A-x-set sa2 -4)
(cap sa2)
(check (= (A-x-get (cap)) -4))
(A-x-set sa2 -7)
(check (= (A-x-get (cap)) -7))

(check (= (A-x-get (ar)) 5))
(ar sa2)
(check (= (A-x-get (ar)) -7))

(x 4)
(check (= (x) 4))

(exit 0)
