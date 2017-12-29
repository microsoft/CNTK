(load-extension "char_constant.so")

(if (and (char? (CHAR-CONSTANT))
	 (string? (STRING-CONSTANT)))
    (exit 0)
    (exit 1))
