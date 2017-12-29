;;; Test out some multi-argument typemaps

(load-extension "./libexample" "scm_init_example_module")

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

; Call the count function
(display (count "Hello World" #\l))
(newline)

; Call the capitalize function
(display (capitalize "hello world"))
(newline)

(exit 0)
