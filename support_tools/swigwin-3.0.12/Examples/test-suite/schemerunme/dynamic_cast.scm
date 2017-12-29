(define f (new-Foo))
(define b (new-Bar))

(define x (Foo-blah f))
(define y (Bar-blah b))

(define a (do-test y))
(if (not (string=? a "Bar::test"))
    (error "Failed!"))

(exit 0)
