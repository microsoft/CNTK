
# for converting TIMIT mlf for use in CNTK

foreach(<>){
    if (/^#/){print; next;} 
    tr/[A-Z]/[a-z]/; 
    if (/^\"/){s,\/,-,g;print; next;}
    @x=split(/\s+/);
    if (@x>=5){$p=$x[4];} 
    if (@x>=3) {$x[2] = $p . "_" . $x[2];} 
    $ln=join(" ",@x);
    print $ln . "\n";
}
