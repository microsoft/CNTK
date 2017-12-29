(define x (new-B))

;; This fails in 1.3a5 because the SWIG/Guile runtime code gets the
;; source and the target of a cast the wrong way around.

(A-hello x)				

(exit 0)
