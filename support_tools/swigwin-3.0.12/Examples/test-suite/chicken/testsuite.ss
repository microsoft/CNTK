(define (lookup-ext-tag tag)
  (cond
    ((equal? tag '(quote swig-contract-assertion-failed))
      '( ((exn type) #f)) )
    (#t '())))

(define-macro (expect-throw tag-form form)
  `(if (condition-case (begin ,form #t)
         ,@(lookup-ext-tag tag-form)
         ((exn) (print "The form threw a different error than expected: " ',form) (exit 1))
         (var () (print "The form did not error as expected: " ',form) (exit 1)))
   (begin (print "The form returned normally when it was expected to throw an error: " ',form) (exit 1))))
