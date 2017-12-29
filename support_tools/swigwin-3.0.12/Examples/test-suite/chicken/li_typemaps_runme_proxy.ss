(require 'li_typemaps)
(load "../schemerunme/li_typemaps_proxy.scm")

(call-with-values (lambda () (inoutr-int2 3 -2))
		  (lambda (a b)
		    (if (not (and (= a 3) (= b -2)))
		      (error "Error in inoutr-int2"))))
(call-with-values (lambda () (out-foo 4))
		  (lambda (a b)
		    (if (not (and (= (slot-ref a 'a) 4) (= b 8)))
		      (error "Error in out-foo"))))

(exit 0)
