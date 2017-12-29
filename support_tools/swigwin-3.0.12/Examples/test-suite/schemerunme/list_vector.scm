(define-macro (check-equality form1 form2)
  `(let ((result1 ,form1)
	 (result2 ,form2))
     (if (not (equal? result1 result2))
	 (error "Check failed:"
		(list 'equal? ',form1 ',form2)
		result1 result2))))

(check-equality (sum-list '(1 3 4 6 7)) 21.0)
(check-equality (sum-vector #(2 4 6 7 9)) 28.0)
(check-equality (one-to-seven-list) '(1 2 3 4 5 6 7))
(check-equality (one-to-seven-vector) #(1 2 3 4 5 6 7))

(check-equality (sum-list2 '(1 3 4 6 7)) 21.0)
(check-equality (sum-vector2 #(2 4 6 7 9)) 28.0)
(check-equality (one-to-seven-list2) '(1 2 3 4 5 6 7))
(check-equality (one-to-seven-vector2) #(1 2 3 4 5 6 7))

(check-equality (sum-lists '(1 2 3) '(4 5 6) '(7 8 9)) 45.0)
(check-equality (sum-lists2 '(1 2 3) '(4 5 6) '(7 8 9)) 45.0)
(check-equality (call-with-values produce-lists list)
	       '(#(0 1 2 3 4)
		 #(0 1 4 9 16)
		 #(0.0 1.5 3.0 4.5 6.0)))

(exit 0)
