;;; Authors: David Beazley <beazley@cs.uchicago.edu>, 1999
;;;          Martin Froehlich <MartinFroehlich@ACM.org>, 2000
;;;
;;; PURPOSE OF THIS FILE: This file is an example for how to use the guile
;;;   scripting options with a little more than trivial script. Example
;;;   derived from David Beazley's matrix evaluation example.  David
;;;   Beazley's annotation: >>Guile script for testing out matrix
;;;   operations. Disclaimer : I'm not a very good scheme
;;;   programmer<<. Martin Froehlich's annotation: >>I'm not a very good
;;;   scheme programmer, too<<.
;;;
;;; Explanation: The three lines at the beginning of this script are
;;; telling the kernel to load the enhanced guile interpreter named
;;; "matrix"; to execute the function "do-test" (-e option) after loading
;;; this script (-s option). There are a lot more options wich allow for
;;; even finer tuning. SEE ALSO: Section "Guile Scripts" in the "Guile
;;; reference manual -- Part I: Preliminaries".
;;;
;;;
;;; This program is distributed in the hope that it will be useful, but
;;; WITHOUT ANY WARRANTY; without even the implied warranty of
;;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

;;; Create a zero matrix

(define (zero M)
  (define (zero-loop M i j)
    (if (< i 4)
	(if (< j 4) (begin
		      (set-m M i j 0.0)
		      (zero-loop M i (+ j 1)))
	    (zero-loop M (+ i 1) 0))))
  (zero-loop M 0 0))

;;; Create an identity matrix

(define (identity M)
  (define (iloop M i)
    (if (< i 4) (begin
		  (set-m M i i 1.0)
		  (iloop M (+ i 1)))))
  (zero M)
  (iloop M 0))

;;; Rotate around x axis

(define (rotx M r)
  (define temp (new-matrix))
  (define rd (/ (* r 3.14159) 180.0))
  (zero temp)
  (set-m temp 0 0 1.0)
  (set-m temp 1 1 (cos rd))
  (set-m temp 1 2 (- 0 (sin rd)))
  (set-m temp 2 1 (sin rd))
  (set-m temp 2 2 (cos rd))
  (set-m temp 3 3 1.0)
  (mat-mult M temp M)
  (destroy-matrix temp))

;;; Rotate around y axis

(define (roty M r)
  (define temp (new-matrix))
  (define rd (/ (* r 3.14159) 180.0))
  (zero temp)
  (set-m temp 1 1 1.0)
  (set-m temp 0 0 (cos rd))
  (set-m temp 0 2 (sin rd))
  (set-m temp 2 0 (- 0 (sin rd)))
  (set-m temp 2 2 (cos rd))
  (set-m temp 3 3 1.0)
  (mat-mult M temp M)
  (destroy-matrix temp))

;;; Rotate around z axis

(define (rotz M r)
  (define temp (new-matrix))
  (define rd (/ (* r 3.14159) 180.0))
  (zero temp)
  (set-m temp 0 0 (cos rd))
  (set-m temp 0 1 (- 0 (sin rd)))
  (set-m temp 1 0 (sin rd))
  (set-m temp 1 1 (cos rd))
  (set-m temp 2 2 1.0)
  (set-m temp 3 3 1.0)
  (mat-mult M temp M)
  (destroy-matrix temp))

;;; Scale a matrix

(define (scale M s)
  (define temp (new-matrix))
  (define (sloop m i s)
    (if (< i 4) (begin
		  (set-m m i i s)
		  (sloop m (+ i 1) s))))
  (zero temp)
  (sloop temp 0 s)
  (mat-mult M temp M)
  (destroy-matrix temp))

;;; Make a matrix with random elements

(define (randmat M)
  (define (rand-loop M i j)
    (if (< i 4)
	(if (< j 4)
            (begin
              (set-m M i j (drand48))
              (rand-loop M i (+ j 1)))
	    (rand-loop M (+ i 1) 0))))
  (rand-loop M 0 0))

;;; stray definitions collected here

(define (rot-test M v t i)
  (if (< i 360) (begin
		  (rotx M 1)
		  (rotz M -0.5)
		  (transform M v t)
		  (rot-test M v t (+ i 1)))))

(define (create-matrix)		; Create some matrices
  (let loop ((i 0) (result '()))
    (if (< i 200)
	(loop (+ i 1) (cons (new-matrix) result))
	result)))

(define (add-mat M ML)
  (define (add-two m1 m2 i j)
    (if (< i 4)
	(if (< j 4)
            (begin
              (set-m m1 i j (+ (get-m m1 i j) (get-m m2 i j)))
              (add-two m1 m2 i (+ j 1)))
	    (add-two m1 m2 (+ i 1) 0))))
  (if (null? ML) *unspecified*
		 (begin
		   (add-two M (car ML) 0 0)
		   (add-mat M (cdr ML)))))

(define (cleanup ML)
  (if (null? ML) *unspecified*
		 (begin
		   (destroy-matrix (car ML))
		   (cleanup (cdr ML)))))

(define (make-random ML)		; Put random values in them
  (if (null? ML) *unspecified*
		 (begin
		   (randmat (car ML))
		   (make-random (cdr ML)))))

(define (mul-mat m ML)
  (if (null? ML) *unspecified*
		 (begin
		   (mat-mult m (car ML) m)
		   (mul-mat m (cdr ML)))))

;;; Now we'll hammer on things a little bit just to make
;;; sure everything works.
(define M1 (new-matrix))		; a matrix
(define v (createv 1 2 3 4))		; a vector
(define t (createv 0 0 0 0))		; the zero-vector
(define M-list (create-matrix))		; get list of marices
(define M (new-matrix))			; yet another matrix

(display "variables defined\n")
(define (do-test x)
  (display "Testing matrix program...\n")

  (identity M1)
  (print-matrix M1)
  (display "Rotate-x 45 degrees\n")
  (rotx M1 45)
  (print-matrix M1)
  (display "Rotate y 30 degrees\n")
  (roty M1 30)
  (print-matrix M1)
  (display "Rotate z 15 degrees\n")
  (rotz M1 15)
  (print-matrix M1)
  (display "Scale 0.5\n")
  (scale M1 0.5)
  (print-matrix M1)

  ;; Rotating ...
  (display "Rotating...\n")
  (rot-test M1 v t 0)
  (printv t)

  (make-random M-list)

  (zero M1)

  (display "Adding them together (in Guile)\n")

  (add-mat M1 M-list)
  (print-matrix M1)

  (display "Doing 200 multiplications (mostly in C)\n")
  (randmat M)

  (mul-mat M M-list)

  (display "Cleaning up\n")

  (cleanup M-list))

