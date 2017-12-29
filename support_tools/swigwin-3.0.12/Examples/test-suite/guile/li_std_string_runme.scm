;; The SWIG modules have "passive" Linkage, i.e., they don't generate
;; Guile modules (namespaces) but simply put all the bindings into the
;; current module.  That's enough for such a simple test.
(dynamic-call "scm_init_li_std_string_module" (dynamic-link "./libli_std_string"))
; Note: when working with non-ascii strings in guile 2
;       Guile doesn't handle non-ascii characters in the default C locale
;       The locale must be set explicitly
;       The setlocale call below takes care of that
;       The locale needs to be a UTF-8 locale to handle the non-ASCII characters
;       But they are named differently on different systems so we try a few until one works

(define (try-set-locale name)
;  (display "testing ")
;  (display name)
;  (display "\n")
  (catch #t
    (lambda ()
      (setlocale LC_ALL name)
      #t
    )
    (lambda (key . parameters)
      #f
    ))
)

(if (not (try-set-locale "C.UTF-8"))     ; Linux
(if (not (try-set-locale "en_US.utf8"))  ; Linux
(if (not (try-set-locale "en_US.UTF-8")) ; Mac OS X
(error "Failed to set any UTF-8 locale")
)))

(load "../schemerunme/li_std_string.scm")
