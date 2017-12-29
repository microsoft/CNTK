;; run with mzscheme -r runme.scm

(load-extension "example.so")

; Call the GCD function

(define x 42)
(define y 105)
(define g (gcd x y))

(display "The gcd of ")
(display x)
(display " and ")
(display y)
(display " is ")
(display g)
(newline)

;  Call the gcdmain() function
(gcdmain #("gcdmain" "42" "105"))


(display (count "Hello World" #\l))
(newline)

(display (capitalize "hello world"))
(newline)
