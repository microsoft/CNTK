(define-macro (check func val test)
  (cons 'begin
    (map
      (lambda (x) 
        `(if (not (,test (,(string->symbol (string-append x func)) ,val) ,val))
           (error ,(string-append "Error in test " x func))))
      (list "in-" "inr-" "out-" "outr-" "inout-" "inoutr-"))))

(define (=~ a b)
  (< (abs (- a b)) 1e-5))
                      
(check "bool" #t and)
(check "int" -2 =)
(check "long" -32 =)
(check "short" -15 =)
(check "uint" 75 =)
(check "ushort" 123 =)
(check "ulong" 462 =)
;(check "uchar" 16 =)
;(check "schar" -53 =)
(check "float" 4.3 =~)
(check "double" -175.42 =~)
(check "longlong" 1634 =)
(check "ulonglong" 6432 =)

;; The checking of inoutr-int2 and out-foo is done in the individual
;; language runme scripts, since chicken returns multiple values
;; and must be checked with call-with-values, while guile just returns a list

;(call-with-values (lambda () (inoutr-int2 3 -2))
;		  (lambda (a b)
;		    (if (not (and (= a 3) (= b -2)))
;		      (error "Error in inoutr-int2"))))
;(call-with-values (lambda () (out-foo 4))
;		  (lambda (a b)
;		    (if (not (and (= (slot-ref a 'a) 4) (= b 8)))
;		      (error "Error in out-foo"))))

;(let ((lst (inoutr-int2 3 -2)))
;  (if (not (and (= (car lst) 3) (= (cadr lst) -2)))
;    (error "Error in inoutr-int2")))

;(let ((lst (out-foo 4)))
;  (if (not (and (= (slot-ref (car lst) 'a) 4) (= (cadr lst) 8)))
;    (error "Error in out-foo")))
