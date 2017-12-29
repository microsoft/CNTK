(load-extension "integers.so")
(require (lib "defmacro.ss"))

(define-macro (throws-exception? form)
  `(with-handlers ((not-break-exn? (lambda (exn) #t)))
     ,form
     #f))

(load "../schemerunme/integers.scm")
