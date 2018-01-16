#!/bin/awk -f
function abs(x) { return ((x < 0.0) ? -x : x); } 

# Set tab as a current separator.
BEGIN { FS="\t" } 

# Record all values from the first file into array a.
NR == FNR { 
    for (i=1; i<=NF; i++)
        a[FNR][i]=$i; 
} 

# During parsing of the second file compare them with remembered
# values using tolerance.
NR != FNR { 
    if($1 != a[FNR][1])
        printf("Sequence key does not match: Baseline %s, Current %s\n", a[FNR][1], $1);

    for (i=2; i<=NF; i++) 
    {
        if (abs($i - a[FNR][i]) >= 0.001 && (abs($i - a[FNR][i])/abs($i)) > 0.03) 
            printf("Line %d, Field %d: Baseline = %f, Current = %f\n", NR, i, a[FNR][i], $i);
    }
}
