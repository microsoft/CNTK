;; feel free to uncomment and comment sections
(load-library 'example "simple.so")

(display "(My-variable): ")
(display (My-variable))
(display "\n")

(display "(My-variable 3.141259): ")
(display (My-variable 3.141259))
(display "\n")

(display "(My-variable): ")
(display (My-variable))
(display "\n")

(display "(fact 5): ")
(display (fact 5))
(display "\n")

(display "(mod 75 7): ")
(display (mod 75 7))
(display "\n")

(display "(get-time): ")
(display (get-time))
(display "\n")

(exit)
