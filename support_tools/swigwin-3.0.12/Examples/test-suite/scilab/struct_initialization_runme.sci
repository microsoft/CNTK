exec("swigtest.start", -1);

if StructC_x_get(instanceC1_get()) <> 10 then swigtesterror(); end

if StructD_x_get(instanceD1_get()) <> 10 then swigtesterror(); end

if StructD_x_get(instanceD2_get()) <> 20 then swigtesterror(); end

if StructD_x_get(instanceD3_get()) <> 30 then swigtesterror(); end

if StructE_x_get(instanceE1_get()) <> 1 then swigtesterror(); end

if StructF_x_get(instanceF1_get()) <> 1 then swigtesterror(); end

exec("swigtest.quit", -1);