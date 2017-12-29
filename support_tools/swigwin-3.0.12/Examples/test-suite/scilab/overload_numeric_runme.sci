exec("swigtest.start", -1);

nums = new_Nums();

// In overloading in Scilab, double has priority over all other numeric types 
checkequal(Nums_over(nums, 0), "double", "Nums_over(nums, 0)");

// Just checkequal if the following are accepted without exceptions being thrown
Nums_doublebounce(nums, %inf);
Nums_doublebounce(nums, -%inf);
Nums_doublebounce(nums, %nan);

delete_Nums(nums);

exec("swigtest.quit", -1);

