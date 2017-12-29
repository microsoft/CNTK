exec("swigtest.start", -1);

// Test constructors for std::deque<int>
intDeque  = new_IntDeque();
intDeque2 = new_IntDeque(3);
intDeque3 = new_IntDeque(4, 42);
//intDeque4 = new_IntDeque(intDeque3);

// Test constructors for std::deque<double>
doubleDeque  = new_DoubleDeque();
doubleDeque2 = new_DoubleDeque(3);
doubleDeque3 = new_DoubleDeque(4, 42.0);
//doubleDeque4 = new_DoubleDeque(doubleDeque3);

// Test constructors for std::deque<Real>
realDeque  = new_RealDeque();
realDeque2 = new_RealDeque(3);
realDeque3 = new_RealDeque(4, 42.0);
//realDeque4 = new_RealDeque(realDeque3);

// average() should return the average of all values in a std::deque<int>
IntDeque_push_back(intDeque, 2);
IntDeque_push_back(intDeque, 4);
IntDeque_push_back(intDeque, 6);
avg = average(intDeque);
checkequal(avg, 4.0, "average(intDeque)");

// half shoud return a deque with elements half of the input elements
RealDeque_clear(realDeque);
RealDeque_push_front(realDeque, 2.0);
RealDeque_push_front(realDeque, 4.0);
halfDeque = half(realDeque);
checkequal(halfDeque, [2., 1.], "half(realDeque)");

// same for halve_in_place
//DoubleDeque_clear(doubleDeque);
//DoubleDeque_push_front(doubleDeque, 2.0);
//DoubleDeque_push_front(doubleDeque, 4.0);
//halfDeque2 = halve_in_place(doubleDeque);
//checkequal(halfDeque2, [2., 1.], "halve_in_place(doubleDeque)");

delete_IntDeque(intDeque);
delete_DoubleDeque(doubleDeque);
delete_RealDeque(realDeque);

exec("swigtest.quit", -1);



