(define f (new-Foo))

(if (not (= (Foo-hola-get f) (Foo-Hello)))
  (error "Error 1"))

(Foo-hola-set f (Foo-Hi))

(if (not (= (Foo-hola-get f) (Foo-Hi)))
  (error "Error 2"))

(Foo-hola-set f (Foo-Hello))

(if (not (= (Foo-hola-get f) (Foo-Hello)))
  (error "Error 3"))

(hi (Hello))

(if (not (= (hi) (Hello)))
  (error "Error 4"))

(exit 0)
