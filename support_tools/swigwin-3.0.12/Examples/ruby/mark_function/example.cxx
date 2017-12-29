#include "example.h"

Animal::Animal(const char* name) : name_(name)
{
}

Animal::~Animal()
{
	name_ = "Destroyed";
}

/* Return the animal's name */
const char* Animal::get_name() const
{
	return name_.c_str();
}
	
Zoo::Zoo()
{
}

Zoo::~Zoo()
{
	return;
}

/* Create a new animal. */
Animal* Zoo::create_animal(const char* name)
{
	return new Animal(name);
}

/* Add a new animal to the zoo. */
void Zoo::add_animal(Animal* animal)
{
	animals.push_back(animal);
}

Animal* Zoo::remove_animal(size_t i)
{
	/* Note a production implementation should check
	   for out of range errors. */
	Animal* result = this->animals[i];
	IterType iter = this->animals.begin();
	std::advance(iter, i);
	this->animals.erase(iter);

	return result;
}

/* Return the number of animals in the zoo. */
size_t Zoo::get_num_animals() const
{
	return animals.size();
}

/* Return a pointer to the ith animal */
Animal* Zoo::get_animal(size_t i) const
{
	return animals[i];
}
