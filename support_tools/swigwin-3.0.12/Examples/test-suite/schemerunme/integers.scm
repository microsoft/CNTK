(define-macro (check-equality form1 form2)
  `(let ((result1 ,form1)
	 (result2 ,form2))
     (if (not (equal? result1 result2))
	 (error "Check failed:"
		(list 'equal? ',form1 ',form2)
		result1 result2))))

(define-macro (check-range function from to)
  `(begin (check-equality (,function ,from) ,from)
	  (check-equality (,function ,to)   ,to)
	  (check-equality (throws-exception? (,function (- ,from 1))) #t)
	  (check-equality (throws-exception? (,function (+ ,to 1))) #t)))

(let ((signed-short-min   (- (expt 2 (- (* (signed-short-size) 8) 1))))
      (signed-short-max   (- (expt 2 (- (* (signed-short-size) 8) 1)) 1))
      (unsigned-short-max (- (expt 2 (* (unsigned-short-size) 8)) 1))
      (signed-int-min     (- (expt 2 (- (* (signed-int-size) 8) 1))))
      (signed-int-max     (- (expt 2 (- (* (signed-int-size) 8) 1)) 1))
      (unsigned-int-max   (- (expt 2 (* (unsigned-int-size) 8)) 1))
      (signed-long-min    (- (expt 2 (- (* (signed-long-size) 8) 1))))
      (signed-long-max    (- (expt 2 (- (* (signed-long-size) 8) 1)) 1))
      (unsigned-long-max  (- (expt 2 (* (unsigned-long-size) 8)) 1))
      (signed-long-long-min    (- (expt 2 (- (* (signed-long-long-size) 8) 1))))
      (signed-long-long-max    (- (expt 2 (- (* (signed-long-long-size) 8) 1)) 1))
      (unsigned-long-long-max  (- (expt 2 (* (unsigned-long-long-size) 8)) 1))
     )

     ;;; signed char, unsigned char typemaps deal with characters, not integers.
     ;; (check-range signed-char-identity (- (expt 2 7)) (- (expt 2 7) 1))
     ;; (check-range unsigned-char-identity 0 (- (expt 2 8) 1))
     (check-range signed-short-identity signed-short-min signed-short-max)
     (check-range unsigned-short-identity 0 unsigned-short-max)
     (check-range signed-int-identity signed-int-min signed-int-max)
     (check-range unsigned-int-identity 0 unsigned-int-max)
     (check-range signed-long-identity signed-long-min signed-long-max)
     (check-range signed-long-long-identity signed-long-long-min signed-long-long-max)

     ;;; unsigned (long) long is broken in guile 1.8 on Mac OS X, skip test
     (if (or (>= (string->number (major-version)) 2)
             (not (equal? (utsname:sysname (uname)) "Darwin")))
         (begin
           (check-range unsigned-long-identity 0 unsigned-long-max)
           (check-range unsigned-long-long-identity 0 unsigned-long-long-max))
     )

)

(exit 0)
