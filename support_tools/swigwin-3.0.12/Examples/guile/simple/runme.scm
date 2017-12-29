
(define (mdisplay-newline . args)       ; does guile-1.3.4 have `format #t'?
  (for-each display args)
  (newline))

(mdisplay-newline (get-time) "My variable = " (My-variable))

(do ((i 0 (1+ i)))
    ((= 14 i))
  (mdisplay-newline i " factorial is " (fact i)))

(define (mods i imax j jmax)
  (if (< i imax)
      (if (< j jmax)
          (begin
            (My-variable (+ (My-variable) (mod i j)))
            (mods i imax (+ j 1) jmax))
          (mods (+ i 1) imax 1 jmax))))

(mods 1 150 1 150)

(mdisplay-newline "My-variable = " (My-variable))

(exit (and (= 1932053504 (fact 13))
           (= 745470.0 (My-variable))))

