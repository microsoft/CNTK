(require 'newobject1)

(define-macro (check-count val)
  `(if (not (= (Foo-fooCount) ,val)) (error "Error checking val " ,val " != " ,(Foo-fooCount))))

(define f (Foo-makeFoo))

(check-count 1)

(define f2 (makeMore f))

(check-count 2)

(set! f #f)
(gc #t)

(check-count 1)

(define f3 (makeMore f2))

(check-count 2)

(set! f3 #f)
(set! f2 #f)

(gc #t)

(check-count 0)

(exit 0)
