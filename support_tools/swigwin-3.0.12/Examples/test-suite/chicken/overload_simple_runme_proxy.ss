(load "overload_simple.so")

(define-macro (check test)
  `(if (not ,test) (error ',test)))

(check (string=? (foo) "foo:"))
(check (string=? (foo 3) "foo:int"))
(check (string=? (foo 3.01) "foo:double"))
(check (string=? (foo "hey") "foo:char *"))

(define f (make <Foo>))
(define b (make <Bar>))
(define b2 (make <Bar> 3))

(check (= (slot-ref b 'num) 0))
(check (= (slot-ref b2 'num) 3))

(check (string=? (foo f) "foo:Foo *"))
(check (string=? (foo b) "foo:Bar *"))
(check (string=? (foo f 3) "foo:Foo *,int"))
(check (string=? (foo 3.2 b) "foo:double,Bar *"))

;; now check blah
(check (string=? (blah 2.01) "blah:double"))
(check (string=? (blah "hey") "blah:char *"))

;; now check spam member functions
(define s (make <Spam>))
(define s2 (make <Spam> 3))
(define s3 (make <Spam> 3.2))
(define s4 (make <Spam> "whee"))
(define s5 (make <Spam> f))
(define s6 (make <Spam> b))

(check (string=? (slot-ref s 'type) "none"))
(check (string=? (slot-ref s2 'type) "int"))
(check (string=? (slot-ref s3 'type) "double"))
(check (string=? (slot-ref s4 'type) "char *"))
(check (string=? (slot-ref s5 'type) "Foo *"))
(check (string=? (slot-ref s6 'type) "Bar *"))

;; now check Spam member functions
(check (string=? (foo s 2) "foo:int"))
(check (string=? (foo s 2.1) "foo:double"))
(check (string=? (foo s "hey") "foo:char *"))
(check (string=? (foo s f) "foo:Foo *"))
(check (string=? (foo s b) "foo:Bar *"))

;; check static member funcs
(check (string=? (Spam-bar 3) "bar:int"))
(check (string=? (Spam-bar 3.2) "bar:double"))
(check (string=? (Spam-bar "hey") "bar:char *"))
(check (string=? (Spam-bar f) "bar:Foo *"))
(check (string=? (Spam-bar b) "bar:Bar *"))

(exit 0)
