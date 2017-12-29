;; run with mzscheme -r runme.scm

(load-extension "example.so")

(display (get-time))

(printf "My-variable = ~a~n" (my-variable))

(let loop ((i 0))
  (when (< i 14) (begin (display i)
			(display " factorial is ")
			(display (fact i))
			(newline)
			(loop (+ i 1)))))

(let loop ((i 1))
  (when (< i 250)
	(begin
	  (let loopi ((j 1))
	    (when (< j 250) (begin (my-variable (+ (my-variable) (mod i j)))
				   (loopi (+ j 1)))))
	  (loop (+ i 1)))))

(printf "My-variable = ~a~n" (my-variable))
