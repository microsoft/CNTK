(define n (fact 4))
(if (not (= n 24))
    (error "Bad return value!"))

(if (not (= (Foo) 42))
    (error "bad variable value!"))

(define t (new-Test))
(if (not (string=? (Test-method t) "Test::method"))
    (error "Bad method return value!"))

(if (not (string=? (do-method t) "Test::method"))
    (error "Bad return value!"))

(if (not (string=? (do-method2 t) "Test::method"))
    (error "Bad return value!"))

(weird "hello" 4)

;; (delete-Test t)

(define t2 (new-Test2))
(define t3 (new-Test3))
(define t4 (new-Test4))
(define t5 (new-Test5))

(if (not (= (foo3 42) 42))
    (error "Bad return value!"))

(if (not (string=? (do-method3 t2 40) "Test2::method"))
    (error "bad return value!"))

(if (not (string=? (do-method3 t3 40) "Test3::method"))
    (error "bad return value"))

(if (not (string=? (do-method3 t4 40) "Test4::method"))
    (error "bad return value"))

(if (not (string=? (do-method3 t5 40) "Test5::method"))
    (error "bad return value"))

(exit 0)
