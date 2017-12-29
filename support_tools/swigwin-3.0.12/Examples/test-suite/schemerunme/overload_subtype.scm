(if (not (= (spam (new-Foo)) 1))
    (error "foo"))

(if (not (= (spam (new-Bar)) 2))
    (error "bar"))

(exit 0)
