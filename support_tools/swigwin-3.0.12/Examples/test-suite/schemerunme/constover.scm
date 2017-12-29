(define p (test "test"))
(if (not (string=? p "test"))
    (error "test failed!"))

(set! p (test-pconst "test"))
(if (not (string=? p "test_pconst"))
    (error "test_pconst failed!"))

(define f (new-Foo))
(set! p (Foo-test f "test"))
(if (not (string=? p "test"))
    (error "member-test failed!"))

(set! p (Foo-test-pconst f "test"))
(if (not (string=? p "test_pconst"))
    (error "member-test_pconst failed!"))

(set! p (Foo-test-constm f "test"))
(if (not (string=? p "test_constmethod"))
    (error "member-test_constm failed!"))

(set! p (Foo-test-pconstm f "test"))
(if (not (string=? p "test_pconstmethod"))
    (error "member-test_pconstm failed!"))

(exit 0)
