/* File : example.i */
%module example

%init{
  zend_printf("This was %%init\n");
}

%minit{
  zend_printf("This was %%minit\n");
}

%mshutdown{
  zend_printf("This was %%shutdown\n");
}

%rinit{
  zend_printf("This was %%rinit\n");
}

%rshutdown{
  zend_printf("This was %%rshutdown\n");
}

%pragma(php) include="include.php";

%pragma(php) code="
# This code is inserted into example.php
echo \"this was php code\\n\";
"

%pragma(php) phpinfo="php_info_print_table_start();"
