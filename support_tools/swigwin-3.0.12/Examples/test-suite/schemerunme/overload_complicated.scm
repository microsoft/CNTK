(define-macro (check form)
  `(if (not ,form)
       (error "Check failed: " ',form)))

(define (=~ a b)
  (< (abs (- a b)) 1e-8))

;; Check first method
(check (=~ (foo 1 2 "bar" 4) 15))

;; Check second method
(check (=~ (foo 1 2) 4811.4))
(check (=~ (foo 1 2 3.2) 4797.2))
(check (=~ (foo 1 2 3.2 #\Q) 4798.2))

(exit 0)
