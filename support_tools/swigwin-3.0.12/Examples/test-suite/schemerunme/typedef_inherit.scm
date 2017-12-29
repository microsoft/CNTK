(define a (new-Foo))
(define b (new-Bar))

(if (not (string=? (do-blah a) "Foo::blah"))
    (error "bad return"))

(if (not (string=? (do-blah b) "Bar::blah"))
    (error "bad return"))

(define c (new-Spam))
(define d (new-Grok))

(if (not (string=? (do-blah2 c) "Spam::blah"))
    (error "bad return"))

(if (not (string=? (do-blah2 d) "Grok::blah"))
    (error "bad return"))

(exit 0)
