#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_

#include <vector>
#include <string>

class Animal
{
protected:
	std::string name_;
public:
	// Construct an animal with a name
	Animal(const char* name);

	// Destruct an animal
	~Animal();

	// Return the animal's name
	const char* get_name() const;
};
	
class Zoo
{
private:
	typedef std::vector<Animal*> AnimalsType;
	typedef AnimalsType::iterator IterType;
protected:
	AnimalsType animals;
public:
	Zoo();
	~Zoo();	

	/* Create a new animal */
	static Animal* create_animal(const char* name);

	/* Add a new animal to the zoo */
	void add_animal(Animal* animal);
	
	/* Remove an animal from the zoo */
	Animal* remove_animal(size_t i);

	/* Return the number of animals in the zoo */
	size_t get_num_animals() const;
	
	/* Return a pointer to the ith animal */
	Animal* get_animal(size_t i) const;
};

#endif /*_EXAMPLE_H_*/
