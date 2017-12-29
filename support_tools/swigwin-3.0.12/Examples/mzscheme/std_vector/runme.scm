;; run with mzscheme -r runme.scm

(load-extension "example.so")

; repeatedly invoke a procedure with v and an index as arguments
(define (with-vector v proc size-proc)
  (let ((size (size-proc v)))
    (define (with-vector-item v i)
      (if (< i size)
          (begin
            (proc v i)
            (with-vector-item v (+ i 1)))))
    (with-vector-item v 0)))

(define (with-intvector v proc)
  (with-vector v proc intvector-length))
(define (with-doublevector v proc)
  (with-vector v proc doublevector-length))

(define (print-doublevector v)
  (with-doublevector v (lambda (v i) (display (doublevector-ref v i)) 
                                     (display " ")))
  (newline))


; Call average with a Scheme list...

(display (average '(1 2 3 4)))
(newline)

; ... or a wrapped std::vector<int>
(define v (new-intvector 4))
(with-intvector v (lambda (v i) (intvector-set! v i (+ i 1))))
(display (average v))
(newline)
(delete-intvector v)

; half will return a Scheme vector.
; Call it with a Scheme vector...

(display (half #(1 1.5 2 2.5 3)))
(newline)

; ... or a wrapped std::vector<double>
(define v (new-doublevector))
(map (lambda (i) (doublevector-push! v i)) '(1 2 3 4))
(display (half v))
(newline)

; now halve a wrapped std::vector<double> in place
(halve-in-place v)
(print-doublevector v)
(delete-doublevector v)

