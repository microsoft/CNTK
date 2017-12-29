;; This file illustrates the proxy C++ interface generated
;; by SWIG.

(load-library 'example "class_proxy.so")
(declare (uses example))
(declare (uses tinyclos))

;; ----- Object creation -----

(display "Creating some objects:\n")
(define c (make <Circle> 10.0))
(display "    Created circle ")
(display c)
(display "\n")
(define s (make <Square> 10.0))
(display "    Created square ")
(display s)
(display "\n")

;; ----- Access a static member -----

(display "\nA total of ")
(display (Shape-nshapes))
(display " shapes were created\n")

;; ----- Member data access -----

;; Set the location of the object

(slot-set! c 'x 20.0)
(slot-set! c 'y 30.0)

(slot-set! s 'x -10.0)
(slot-set! s 'y 5.0)

(display "\nHere is their current position:\n")
(display "    Circle = (")
(display (slot-ref c 'x))
(display ", ")
(display (slot-ref c 'y))
(display ")\n")
(display "    Square = (")
(display (slot-ref s 'x))
(display ", ")
(display (slot-ref s 'y))
(display ")\n")

;; ----- Call some methods -----

(display "\nHere are some properties of the shapes:\n")
(let
    ((disp (lambda (o)
             (display "   ")
             (display o)
             (display "\n")
             (display "        area      = ")
             (display (area o))
             (display "\n")
             (display "        perimeter = ")
             (display (perimeter o))
             (display "\n"))))
  (disp c)
  (disp s))

(display "\nGuess I'll clean up now\n")

;; Note: Invoke the virtual destructors by forcing garbage collection
(set! c 77)
(set! s 88)
(gc #t)

(display (Shape-nshapes))
(display " shapes remain\n")
(display "Goodbye\n")

(exit)
