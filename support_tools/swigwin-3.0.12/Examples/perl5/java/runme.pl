use example;

example::JvCreateJavaVM(undef);
example::JvAttachCurrentThread(undef, undef);

$e1 = new example::Example(1);
print $e1->{mPublicInt},"\n";

$e2 = new example::Example(2);
print $e2->{mPublicInt},"\n";

$i = $e1->Add(1,2);
print $i,"\n";

$d = $e2->Add(1.0,2.0);
print $d,"\n";

$d = $e2->Add("1","2");
print $d,"\n";

$e3 = $e1->Add($e1,$e2);
print $e3->{mPublicInt},"\n";


$s = $e2->Add("a","b");
print $s,"\n";


example::JvDetachCurrentThread()
