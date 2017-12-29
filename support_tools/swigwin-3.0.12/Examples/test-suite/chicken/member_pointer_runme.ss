(require 'member_pointer)

(define (check-eq? msg expected actual)
  (if (not (= expected actual))
    (error "Error " msg ": expected " expected " got " actual)))

(define area-pt (areapt))
(define perim-pt (perimeterpt))

(define s (new-Square 10))

(check-eq? "Square area" 100.0 (do-op s area-pt))
(check-eq? "Square perim" 40.0 (do-op s perim-pt))

(check-eq? "Square area" 100.0 (do-op s (areavar)))
(check-eq? "Square perim" 40.0 (do-op s (perimetervar)))

;; Set areavar to return value of function
(areavar perim-pt)
(check-eq? "Square perim" 40 (do-op s (areavar)))

(check-eq? "Square area" 100.0 (do-op s (AREAPT)))
(check-eq? "Square perim" 40.0 (do-op s (PERIMPT)))

(define test (NULLPT))

(perimetervar (AREAPT))
(check-eq? "Square area" 100.0 (do-op s (perimetervar)))
