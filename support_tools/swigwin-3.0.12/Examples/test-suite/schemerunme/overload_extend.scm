(define f (new-Foo))

(if (not (= (Foo-test f 3) 1))
    (error "test integer bad"))

(if (not (= (Foo-test f "hello") 2))
    (error "test string bad"))

(if (not (= (Foo-test f 3.5 2.5) 6.0))
    (error "test reals bad"))

(exit 0)
