(if (and (char? (CHAR-CONSTANT))
	 (string? (STRING-CONSTANT)))
    (exit 0)
    (exit 1))
