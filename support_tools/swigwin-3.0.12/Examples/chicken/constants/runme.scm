;; feel free to uncomment and comment sections

(load-library 'example "./constants.so")

(display "starting test ... you will see 'finished' if successful.\n")
(or (= (ICONST) 42) (exit 1))
(or (< (abs (- (FCONST) 2.1828)) 0.00001) (exit 1))
(or (char=? (CCONST) #\x) (exit 1))
(or (char=? (CCONST2) #\newline) (exit 1))
(or (string=? (SCONST) "Hello World") (exit 1))
(or (string=? (SCONST2) "\"Hello World\"") (exit 1))
(or (< (abs (- (EXPR) (+ (ICONST) (* 3 (FCONST))))) 0.00001) (exit 1))
(or (= (iconstX) 37) (exit 1))
(or (< (abs (- (fconstX) 3.14)) 0.00001) (exit 1))
(display "finished test.\n")
(exit 0)
