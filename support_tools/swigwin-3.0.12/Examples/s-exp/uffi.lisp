;;; This is experimental code that uses the s-expression
;;; representation of a C/C++ library interface to generate Foreign
;;; Function Interface definitions for use with Kevin Rosenberg's
;;; UFFI.
;;;
;;; Written by Matthias Koeppe <mkoeppe@mail.math.uni-magdeburg.de>

(eval-when (:compile-toplevel :load-toplevel :execute)
  (require 'port)				; from CLOCC
  (require 'uffi))

(in-package :cl-user)

;; Interaction with the SWIG binary

(defvar *swig-source-directory* #p"/home/mkoeppe/s/swig1.3/")

(defvar *swig-program* (merge-pathnames "swig" *swig-source-directory*))

(defun run-swig (swig-interface-file-name &key directory-search-list module
		 ignore-errors c++)
  (let ((temp-file-name "/tmp/swig.lsp"))
    (let ((process
	   (port:run-prog (namestring *swig-program*)
			  :output t
			  :args `(,@(and c++ '("-c++"))
				  "-sexp"
				  ,@(mapcar (lambda (dir)
				       (concatenate 'string
						    "-I" (namestring dir)))
					    directory-search-list)
				  ,@(and module
					 `("-module" ,module))
				  "-o" ,temp-file-name
				  ,(namestring swig-interface-file-name)))))
      #+cmu (unless (or (zerop (ext:process-exit-code process))
			ignore-errors)
	      (error "Process swig exited abnormally"))
      (with-open-file (s temp-file-name)
	(read s)))))

;; Type system

(defun parse-swigtype (type-string &key start end junk-ok)
  "Parse TYPE-STRING as SWIG's internal representation of C/C++
types. Return two values: The type description (an improper list) and
the terminating index into TYPE-STRING."
  ;; SWIG's internal representation is described in Source/Swig/stype.c
  (unless start
    (setq start 0))
  (unless end
    (setq end (length type-string)))
  (flet ((prefix-match (prefix)
	   (let ((position (mismatch prefix type-string :start2 start :end2 end)))
	     (or (not position)
		 (= position (length prefix)))))
	 (bad-type-error (reason)
	   (error "Bad SWIG type (~A): ~A" reason
		  (subseq type-string start end)))
	 (type-char (index)
	   (and (< index (length type-string))
		(char type-string index)))	       
	 (cons-and-recurse (prefix start end)
	   (multiple-value-bind (type-description index)
	       (parse-swigtype type-string :start start :end end
				:junk-ok junk-ok)
	     (values (cons prefix type-description)
		     index))))
    (cond
      ((prefix-match "p.")		; pointer
       (cons-and-recurse '* (+ start 2) end))
      ((prefix-match "r.")		; C++ reference
       (cons-and-recurse '& (+ start 2) end))
      ((prefix-match "a(")		; array
       (let ((closing-paren (position #\) type-string
				  :start (+ start 2)
				  :end end)))
	 (unless closing-paren
	   (bad-type-error "missing right paren"))
	 (unless (eql (type-char (+ closing-paren 1)) #\.)
	   (bad-type-error "missing dot"))
	 (cons-and-recurse (list 'ARRAY (subseq type-string (+ start 2) closing-paren))
			   (+ closing-paren 2) end)))
      ((prefix-match "q(")		; qualifier (const, volatile)
       (let ((closing-paren (position #\) type-string
				  :start (+ start 2)
				  :end end)))
	 (unless closing-paren
	   (bad-type-error "missing right paren"))
	 (unless (eql (type-char (+ closing-paren 1)) #\.)
	   (bad-type-error "missing dot"))
	 (cons-and-recurse (list 'QUALIFIER (subseq type-string (+ start 2) closing-paren))
			   (+ closing-paren 2) end)))
      ((prefix-match "m(")		; C++ member pointer
       (multiple-value-bind (class-type class-end-index)
	   (parse-swigtype type-string :junk-ok t
			    :start (+ start 2) :end end)
	 (unless (eql (type-char class-end-index) #\))
	   (bad-type-error "missing right paren"))
	 (unless (eql (type-char (+ class-end-index 1)) #\.)
	   (bad-type-error "missing dot"))
	 (cons-and-recurse (list 'MEMBER-POINTER class-type)
			   (+ class-end-index 2) end)))	 
      ((prefix-match "f(")		; function
       (loop with index = (+ start 2) 
	     until (eql (type-char index) #\))
	     collect (multiple-value-bind (arg-type arg-end-index)
			 (parse-swigtype type-string :junk-ok t
					  :start index :end end)
		       (case (type-char arg-end-index)
			 (#\, (setq index (+ arg-end-index 1)))
			 (#\) (setq index arg-end-index))
			 (otherwise (bad-type-error "comma or right paren expected"))) 
		       arg-type)
	     into arg-types
	     finally (unless (eql (type-char (+ index 1)) #\.)
		       (bad-type-error "missing dot"))
	     (return (cons-and-recurse (cons 'FUNCTION arg-types)
				       (+ index 2) end))))
      ((prefix-match "v(")		;varargs
       (let ((closing-paren (position #\) type-string
				  :start (+ start 2)
				  :end end)))
	 (unless closing-paren
	   (bad-type-error "missing right paren"))
	 (values (list 'VARARGS (subseq type-string (+ start 2) closing-paren))
		 (+ closing-paren 1))))
      (t (let ((junk-position (position-if (lambda (char)
					     (member char '(#\, #\( #\) #\.)))
					   type-string
					   :start start :end end)))
	   (cond (junk-position		; found junk
		  (unless junk-ok
		    (bad-type-error "trailing junk"))
		  (values (subseq type-string start junk-position)
			  junk-position))
		 (t
		  (values (subseq type-string start end)
			  end))))))))

(defun swigtype-function-p (swigtype)
  "Check whether SWIGTYPE designates a function.  If so, the second
value is the list of argument types, and the third value is the return
type."
  (if (and (consp swigtype)
	   (consp (first swigtype))
	   (eql (first (first swigtype)) 'FUNCTION))
      (values t (rest (first swigtype)) (rest swigtype))
      (values nil nil nil)))
	      

;; UFFI

(defvar *uffi-definitions* '())

(defconstant *uffi-default-primitive-type-alist*
  '(("char" . :char)
    ("unsigned char" . :unsigned-byte)
    ("signed char" . :byte)
    ("short" . :short)
    ("signed short" . :short)
    ("unsigned short" . :unsigned-short)
    ("int" . :int)
    ("signed int" . :int)
    ("unsigned int" . :unsigned-int)
    ("long" . :long)
    ("signed long" . :long)
    ("unsigned long" . :unsigned-long)
    ("float" . :float)
    ("double" . :double)
    ((* . "char") . :cstring)
    ((* . "void") . :pointer-void)
    ("void" . :void)))

(defvar *uffi-primitive-type-alist* *uffi-default-primitive-type-alist*)

(defun uffi-type-spec (type-list)
  "Return the UFFI type spec equivalent to TYPE-LIST, or NIL if there
is no representation."
  (let ((primitive-type-pair
	 (assoc type-list *uffi-primitive-type-alist* :test 'equal)))
    (cond
      (primitive-type-pair
       (cdr primitive-type-pair))
      ((and (consp type-list)
	    (eql (first type-list) '*))
       (let ((base-type-spec (uffi-type-spec (rest type-list))))
	 (cond
	   ((not base-type-spec)
	    :pointer-void)
	   (t
	    (list '* base-type-spec)))))
      (t nil))))

;; Parse tree

(defvar *uffi-output* nil)

(defun emit-uffi-definition (uffi-definition)
  (format *uffi-output* "~&~S~%" uffi-definition)
  (push uffi-definition *uffi-definitions*))

(defun make-cl-symbol (c-identifier &key uninterned)
  (let ((name (substitute #\- #\_ (string-upcase c-identifier))))
    (if uninterned
	(make-symbol name)
	(intern name))))

(defvar *class-scope* '() "A stack of names of nested C++ classes.")

(defvar *struct-fields* '())

(defvar *linkage* :C "NIL or :C")

(defgeneric handle-node (node-type &key &allow-other-keys)
  (:documentation "Handle a node of SWIG's parse tree of a C/C++ program"))

(defmethod handle-node ((node-type t) &key &allow-other-keys)
  ;; do nothing for unknown node types
  nil)

(defmethod handle-node ((node-type (eql 'cdecl)) &key name decl storage parms type &allow-other-keys)
  (let ((swigtype (parse-swigtype (concatenate 'string decl type))))
    (let ((*print-pretty* nil) ; or FUNCTION would be printed as #' by cmucl
	  (*print-circle* t))
      (format *uffi-output* "~&;; C Declaration: ~A ~A ~A ~A~%;;  with-parms ~W~%;;   of-type ~W~%"
	      storage type name decl parms swigtype))
    (multiple-value-bind (function-p arg-swigtype-list return-swigtype)
	(swigtype-function-p swigtype)
      (declare (ignore arg-swigtype-list))
      (cond
	((and (null *class-scope*) function-p
	      (or (eql *linkage* :c)
		  (string= storage "externc")))
	 ;; ordinary top-level function with C linkage
	 (let ((argnum 0)
	       (argname-list '()))
	   (flet ((unique-argname (name)
		    ;; Sometimes the functions in SWIG interfaces
		    ;; do not have unique names.  Make them unique
		    ;; by adding a suffix.  Also avoid symbols
		    ;; that are specially bound.
		    (unless name
		      (setq name (format nil "arg~D" argnum)))
		    (let ((argname (make-cl-symbol name)))
		      (when (boundp argname) ;specially bound
			(setq argname (make-cl-symbol name :uninterned t)))
		      (push argname argname-list)
		      argname)))
	     (let ((uffi-arg-list
		    (mapcan (lambda (param)
			      (incf argnum)
			      (destructuring-bind (&key name type &allow-other-keys) param
				(let ((uffi-type (uffi-type-spec (parse-swigtype type))))
				  (cond
				    ((not uffi-type)
				     (format *uffi-output* "~&;; Warning: Cannot handle type ~S of argument `~A'~%"
					     type name)
				     (return-from handle-node))
				    ((eq uffi-type :void)
				     '())
				    (t
				     (let ((symbol (unique-argname name)))
				       (list `(,symbol ,uffi-type))))))))
			    parms))
		   (uffi-return-type
		    (uffi-type-spec return-swigtype)))
	       (unless uffi-return-type
		 (format *uffi-output* "~&;; Warning: Cannot handle return type `~S'~%"
			 return-swigtype)
		 (return-from handle-node))
	       (emit-uffi-definition `(UFFI:DEF-FUNCTION ,name ,uffi-arg-list :RETURNING ,uffi-return-type))))))
	((and (not (null *class-scope*)) (null (rest *class-scope*))
	      (not function-p))	; class/struct member (no nested structs)
	 (let ((uffi-type (uffi-type-spec swigtype)))
	   (unless  uffi-type
	     (format *uffi-output* "~&;; Warning: Cannot handle type ~S of struct field `~A'~%"
		     type name)
	     (return-from handle-node))
	   (push `(,(make-cl-symbol name) ,uffi-type) *struct-fields*)))))))

(defmethod handle-node ((node-type (eql 'class)) &key name children kind &allow-other-keys)
  (format *uffi-output* "~&;; Class ~A~%" name)
  (let ((*class-scope* (cons name *class-scope*))
	(*struct-fields* '()))
    (dolist (child children)
      (apply 'handle-node child))
    (emit-uffi-definition `(,(if (string= kind "union")
				 'UFFI:DEF-UNION
				 'UFFI:DEF-STRUCT)
			    ,(make-cl-symbol name) ,@(nreverse *struct-fields*)))))

(defmethod handle-node ((node-type (eql 'top)) &key children &allow-other-keys)
  (dolist (child children)
    (apply 'handle-node child)))
  
(defmethod handle-node ((node-type (eql 'include)) &key name children &allow-other-keys)
  (format *uffi-output* ";; INCLUDE ~A~%" name)
  (dolist (child children)
    (apply 'handle-node child)))

(defmethod handle-node ((node-type (eql 'extern)) &key name children &allow-other-keys)
  (format *uffi-output* ";; EXTERN \"C\" ~A~%" name)
  (let ((*linkage* :c))
    (dolist (child children)
      (apply 'handle-node child))))

;;(defun compute-uffi-definitions (swig-interface)
;;  (let ((*uffi-definitions* '()))
;;    (handle-node swig-interface)
;;    *uffi-definitions*))

;; Test instances

;;; Link to SWIG itself

#||

(defparameter *c++-compiler* "g++")

(defun stdc++-library (&key env)
  (let ((error-output (make-string-output-stream)))
    (let ((name-output (make-string-output-stream)))
      (let ((proc (ext:run-program
		   *c++-compiler*
		   '("-print-file-name=libstdc++.so")
		   :env env
		   :input nil
		   :output name-output
		   :error error-output)))
	(unless proc
	  (error "Could not run ~A" *c++-compiler*))
	(unless (zerop (ext:process-exit-code proc))
	  (system:serve-all-events 0)
	  (error "~A failed:~%~A" *c++-compiler*
		 (get-output-stream-string error-output))))
      (string-right-trim '(#\Newline) (get-output-stream-string name-output)))))

(defvar *swig-interface* nil)

(defvar *swig-uffi-pathname* #p"/tmp/swig-uffi.lisp")

(defun link-swig ()
  (setq *swig-interface*
	(run-swig (merge-pathnames "Source/swig.i" *swig-source-directory*)
		  :directory-search-list
		  (list (merge-pathnames "Source/" *swig-source-directory*))
		  :module "swig"
		  :ignore-errors t
		  :c++ t))
  (with-open-file (f *swig-uffi-pathname* :direction :output)
    (let ((*linkage* :c++)
	  (*uffi-definitions* '())
	  (*uffi-output* f)
	  (*uffi-primitive-type-alist* *uffi-default-primitive-type-alist*))
      (apply 'handle-node *swig-interface*)))
  (compile-file *swig-uffi-pathname*)
  (alien:load-foreign (merge-pathnames "Source/libswig.a"
				       *swig-source-directory*)
		      :libraries (list (stdc++-library)))
  ;; FIXME: UFFI stuffes a "-l" in front of the passed library names
  ;;  (uffi:load-foreign-library (merge-pathnames "Source/libswig.a"
  ;;                                              *swig-source-directory*)
  ;;                             :supporting-libraries
  ;;                             (list (stdc++-library)))
  (load (compile-file-pathname *swig-uffi-pathname*)))

||#

;;;; TODO:

;; * How to do type lookups?  Is everything important that SWIG knows
;;   about the types written out?  What to make of typemaps?
;;
;; * Wrapped functions should probably automatically COERCE their
;;   arguments (as of type DOUBLE-FLOAT), to make the functions more
;;   flexible?
;;
;; * Why are the functions created by FFI interpreted?
;;
;; * We can't deal with more complicated structs and C++ classes
;; directly with the FFI; we have to emit SWIG wrappers that access
;; those classes.
;;
;; * A CLOS layer where structure fields are mapped as slots.  It
;; looks like we need MOP functions to implement this.
;;
;; * Maybe modify SWIG so that key-value hashes are distinguished from
;; value-value hashes.
