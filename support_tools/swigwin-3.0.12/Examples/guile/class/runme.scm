; file: runme.scm

; This file illustrates the proxy class C++ interface generated
; by SWIG.

(load-extension "./libexample" "scm_init_example_module")

; Convenience wrapper around the display function
; (which only accepts one argument at the time)

(define (mdisplay-newline . args)
  (for-each display args)
  (newline))

; ----- Object creation -----

(mdisplay-newline "Creating some objects:")
(define c (new-Circle 10))
(mdisplay-newline "    Created circle " c)
(define s (new-Square 10))
(mdisplay-newline "    Created square " s)

; ----- Access a static member -----

(mdisplay-newline "\nA total of " (Shape-nshapes) " shapes were created")

; ----- Member data access -----

; Set the location of the object

(Shape-x-set c 20)
(Shape-y-set c 30)

(Shape-x-set s -10)
(Shape-y-set s 5)

(mdisplay-newline "\nHere is their current position:")
(mdisplay-newline "    Circle = (" (Shape-x-get c) "," (Shape-y-get c) ")")
(mdisplay-newline "    Square = (" (Shape-x-get s) "," (Shape-y-get s) ")")

; ----- Call some methods -----

(mdisplay-newline "\nHere are some properties of the shapes:")
(define (shape-props o)
  (mdisplay-newline "   " o)
  (mdisplay-newline "   area      = " (Shape-area o))
  (mdisplay-newline "   perimeter = " (Shape-perimeter o)))
(for-each  shape-props (list c s))

(mdisplay-newline "\nGuess I'll clean up now")

; Note: this invokes the virtual destructor
(delete-Shape c)
(delete-Shape s)

(define s 3)
(mdisplay-newline (Shape-nshapes) " shapes remain")
(mdisplay-newline "Goodbye")

(exit 0)
