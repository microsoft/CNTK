(define a (new-Bar))

(if (not (string=? (Bar-blah a) "Bar::blah"))
    (error "Wrong string"))

(exit 0)
