
(load-extension "./libexample" "scm_init_example_module")

; repeatedly invoke a procedure with v and an index as arguments
(define (with-vector v proc size-proc)
  (let ((size (size-proc v)))
    (define (with-vector-item v i)
      (if (< i size)
          (begin
            (proc v i)
            (with-vector-item v (+ i 1)))))
    (with-vector-item v 0)))

(define (with-IntVector v proc)
  (with-vector v proc IntVector-length))
(define (with-DoubleVector v proc)
  (with-vector v proc DoubleVector-length))

(define (print-DoubleVector v)
  (with-DoubleVector v (lambda (v i) (display (DoubleVector-ref v i)) 
                                     (display " ")))
  (newline))


; Call average with a Scheme list...

(display (average '(1 2 3 4)))
(newline)

; ... or a wrapped std::vector<int>
(define v (new-IntVector 4))
(with-IntVector v (lambda (v i) (IntVector-set! v i (+ i 1))))
(display (average v))
(newline)
(delete-IntVector v)

; half will return a Scheme vector.
; Call it with a Scheme vector...

(display (half #(1 1.5 2 2.5 3)))
(newline)

; ... or a wrapped std::vector<double>
(define v (new-DoubleVector))
(map (lambda (i) (DoubleVector-push! v i)) '(1 2 3 4))
(display (half v))
(newline)

; now halve a wrapped std::vector<double> in place
(halve-in-place v)
(print-DoubleVector v)
(delete-DoubleVector v)

(exit 0)
