;; This file illustrates the low-level C++ interface generated
;; by SWIG.

(load-library 'example "class.so")
(declare (uses example))

;; ----- Object creation -----

(display "Creating some objects:\n")
(define c (new-Circle 10.0))
(display "    Created circle ")
(display c)
(display "\n")
(define s (new-Square 10.0))
(display "    Created square ")
(display s)
(display "\n")

;; ----- Access a static member -----

(display "\nA total of ")
(display (Shape-nshapes))
(display " shapes were created\n")

;; ----- Member data access -----

;; Set the location of the object

(Shape-x-set c 20.0)
(Shape-y-set c 30.0)

(Shape-x-set s -10.0)
(Shape-y-set s 5.0)

(display "\nHere is their current position:\n")
(display "    Circle = (")
(display (Shape-x-get c))
(display ", ")
(display (Shape-y-get c))
(display ")\n")
(display "    Square = (")
(display (Shape-x-get s))
(display ", ")
(display (Shape-y-get s))
(display ")\n")

;; ----- Call some methods -----

(display "\nHere are some properties of the shapes:\n")
(let
    ((disp (lambda (o)
             (display "   ")
             (display o)
             (display "\n")
             (display "        area      = ")
             (display (Shape-area o))
             (display "\n")
             (display "        perimeter = ")
             (display (Shape-perimeter o))
             (display "\n"))))
  (disp c)
  (disp s))

(display "\nGuess I'll clean up now\n")

;; Note: this invokes the virtual destructor
(set! c #f)
(set! s #f)
(gc #t)

(set! s 3)
(display (Shape-nshapes))
(display " shapes remain\n")
(display "Goodbye\n")

(exit)
