(define a (new-Foo))
(define b (new-Bar))
(define c (new-Spam))

(if (not (string=? (do-blah a) "Foo::blah"))
    (error "Bad return"))

(if (not (string=? (do-blah b) "Bar::blah"))
    (error "Bad return"))

(if (not (string=? (do-blah c) "Spam::blah"))
    (error "bad return"))

(exit 0)
