;;;; Automatic test of multiple return values

(let ((quotient/remainder (divide-l 37 5)))
  (if (not (equal? quotient/remainder '(7 2)))
      (exit 1)))

(let ((quotient-remainder-vector (divide-v 41 7)))
  (if (not (equal? quotient-remainder-vector #(5 6)))
      (exit 1)))

(call-with-values (lambda ()
		    (divide-mv 91 13))
		  (lambda (quotient remainder)
		    (if (not (and (= quotient 7)
				  (= remainder 0)))
			(exit 1))))

(exit 0)
			     
