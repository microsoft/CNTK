;; This file demonstrates the overloading capabilities of SWIG

(load-library 'example "overload.so")

;; Low level
;; ---------

(display "
Trying low level code ...
  (foo 1)
  (foo \"some string\")
  (define A-FOO (new-Foo))
  (define ANOTHER-FOO (new-Foo A-FOO)) ;; copy constructor
  (Foo-bar A-FOO 2)
  (Foo-bar ANOTHER-FOO \"another string\" 3)
")

(primitive:foo 1)
(primitive:foo "some string")
(define A-FOO (slot-ref (primitive:new-Foo) 'swig-this))
(define ANOTHER-FOO (slot-ref (primitive:new-Foo A-FOO) 'swig-this)) ;; copy constructor
(primitive:Foo-bar A-FOO 2)
(primitive:Foo-bar ANOTHER-FOO "another string" 3)

;; TinyCLOS
;; --------

(display "
Trying TinyCLOS code ...
  (+foo+ 1)
  (+foo+ \"some string\")
  (define A-FOO (make <Foo>))
  (define ANOTHER-FOO (make <Foo> A-FOO)) ;; copy constructor
  (-bar- A-FOO 2)
  (-bar- ANOTHER-FOO \"another string\" 3)
")

(foo 1)
(foo "some string")
(define A-FOO (make <Foo>))
(define ANOTHER-FOO (make <Foo> A-FOO)) ;; copy constructor
(bar A-FOO 2)
(bar ANOTHER-FOO "another string" 3)

(exit)
