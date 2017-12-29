(define-macro (check form)
  `(if (not ,form)
       (error "Check failed: " ',form)))

(define p (produce-int-pointer 47 11))

(check (= (consume-int-pointer p) 47))

(define q (frobnicate-int-pointer p))

(check (= (consume-int-pointer q) 11))

(exit 0)
