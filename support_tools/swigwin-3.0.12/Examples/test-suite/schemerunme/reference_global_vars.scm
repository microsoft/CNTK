(define (!= a b) (not (= a b)))

; const class reference variable
(if (!= (TestClass-num-get (getconstTC)) 33)
  (begin (display "Runtime test 1 failed.\n") (exit 1)))

; primitive reference variables
(var-bool (createref-bool #f))
(if (value-bool (var-bool))
  (begin (display "Runtime test 2 failed.\n") (exit 1)))

(var-char (createref-char #\w))
(if (not (char=? (value-char (var-char)) #\w))
  (begin (display "Runtime test 3 failed.\n") (exit 1)))

(var-unsigned-char (createref-unsigned-char #\newline))
(if (not (char=? (value-unsigned-char (var-unsigned-char)) #\newline))
  (begin (display "Runtime test 4 failed.\n") (exit 1)))

(var-signed-char (createref-signed-char #\newline))
(if (not (char=? (value-signed-char (var-signed-char)) #\newline))
  (begin (display "Runtime test 5 failed.\n") (exit 1)))

(var-unsigned-short (createref-unsigned-short 10))
(if (!= (value-unsigned-short (var-unsigned-short)) 10)
  (begin (display "Runtime test 6 failed.\n") (exit 1)))

(var-int (createref-int 10))
(if (!= (value-int (var-int)) 10)
  (begin (display "Runtime test 7 failed.\n") (exit 1)))

(var-unsigned-int (createref-unsigned-int 10))
(if (!= (value-unsigned-int (var-unsigned-int)) 10)
  (begin (display "Runtime test 8 failed.\n") (exit 1)))

(var-long (createref-long 10))
(if (!= (value-long (var-long)) 10)
  (begin (display "Runtime test 9 failed.\n") (exit 1)))

(var-unsigned-long (createref-unsigned-long 10))
(if (!= (value-unsigned-long (var-unsigned-long)) 10)
  (begin (display "Runtime test 10 failed.\n") (exit 1)))

;skip long long and unsigned long long

(var-float (createref-float 10.5))
(if (!= (value-float (var-float)) 10.5)
  (begin (display "Runtime test 11 failed.\n") (exit 1)))

(var-double (createref-double 10.55))
(if (!= (value-double (var-double)) 10.55)
  (begin (display "Runtime test 12 failed.\n") (exit 1)))

;class reference
(var-TestClass (createref-TestClass (new-TestClass 20)))
(if (!= (TestClass-num-get (value-TestClass (var-TestClass))) 20)
  (begin (display "Runtime test 13 failed.\n") (exit 1)))

(exit 0)
