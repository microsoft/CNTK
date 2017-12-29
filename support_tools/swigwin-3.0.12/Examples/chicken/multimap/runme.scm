;; feel free to uncomment and comment sections

(load-library 'example "multimap.so")

(display "(gcd 90 12): ")
(display (gcd 90 12))
(display "\n")

(display "(circle 0.5 0.5): ")
(display (circle 0.5 0.5))
(display "\n")

(display "(circle 1.0 1.0): ")
(handle-exceptions exvar
  (if (= (car exvar) 9)
    (display "success: exception thrown")
    (display "an incorrect exception was thrown"))
  (begin
    (circle 1.0 1.0)
    (display "an exception was not thrown when it should have been")))
(display "\n")

(display "(circle 1 1): ")
(handle-exceptions exvar
  (if (= (car exvar) 9)
    (display "success: exception thrown")
    (display "an incorrect exception was thrown"))
  (begin
    (circle 1 1)
    (display "an exception was not thrown when it should have been")))
(display "\n")

(display "(capitalize \"will this be all capital letters?\"): ")
(display (capitalize "will this be all capital letters?"))
(display "\n")

(display "(count \"jumpity little spider\" #\\t): ")
(display (count "jumpity little spider" #\t))
(display "\n")

(display "(gcdmain '#(\"hi\" \"there\")): ")
(display (gcdmain '#("hi" "there")))
(display "\n")

(display "(gcdmain '#(\"gcd\" \"9\" \"28\")): ")
(gcdmain '#("gcd" "9" "28"))
(display "\n")

(display "(gcdmain '#(\"gcd\" \"12\" \"90\")): ")
(gcdmain '#("gcd" "12" "90"))
(display "\n")

(display "squarecubed 3: ")
(call-with-values (lambda() (squareCubed 3)) 
		  (lambda (a b) (printf "~A ~A" a b)))
(display "\n")

(exit)
