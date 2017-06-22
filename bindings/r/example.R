library(cntk)
library(magrittr)

op_minus(c(1, 2, 3), c(4, 5, 6)) %>% eval

x <- op_input_variable(2)
y <- op_input_variable(2)
x0 <- matrix(c(2.0, 1.0), ncol = 2)
y0 <- matrix(c(4.0, 6.0), ncol = 2)
args <- dict(x = x0, y = y0)
loss_squared_error(x, y) %>% func_eval(args)

mat = matrix(c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), nrow = 3)
constant_data = op_constant(mat)
constant_data$value
