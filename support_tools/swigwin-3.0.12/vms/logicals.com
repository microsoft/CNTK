$!
$!
$!
$ proc = f$environment("PROCEDURE")
$ proc = f$parse(proc,"sys$disk:[]",,,"NO_CONCEAL")
$ cur_dev = f$parse(proc,,,"DEVICE","SYNTAX_ONLY")
$ cur_dir = f$parse(proc,,,"DIRECTORY","SYNTAX_ONLY")
$ cur_dir = f$extract(1,f$length(cur_dir)-2,cur_dir)
$ cur_dir = cur_dir - "["
$ cur_dir = cur_dir - "]"
$ cur_dir = cur_dir - "<"
$ cur_dir = cur_dir - ">"
$
$! remove trealing .VMS
$ root_dir = f$extract(0,f$length(cur_dir)-4,cur_dir)
$
$ define 'p1' /trans=concealed swig_root 'cur_dev'['root_dir'.]

