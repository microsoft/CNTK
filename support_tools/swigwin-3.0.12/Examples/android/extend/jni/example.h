/* File : example.h */

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>

struct Streamer {
  virtual void display(std::string text) const = 0;
  virtual ~Streamer() {}
};
void setStreamer(Streamer* streamer);
Streamer& getStreamer();

template<typename T> Streamer& operator<<(Streamer& stream, T const& val) {
  std::ostringstream s;
  s << val;
  stream.display(s.str());
  return stream;
}

class Employee {
private:
	std::string name;
public:
	Employee(const char* n): name(n) {}
	virtual std::string getTitle() { return getPosition() + " " + getName(); }
	virtual std::string getName() { return name; }
	virtual std::string getPosition() const { return "Employee"; }
	virtual ~Employee() { getStreamer() << "~Employee() @ " << this << "\n"; }
};


class Manager: public Employee {
public:
	Manager(const char* n): Employee(n) {}
	virtual std::string getPosition() const { return "Manager"; }
};


class EmployeeList {
	std::vector<Employee*> list;
public:
	EmployeeList() {
		list.push_back(new Employee("Bob"));
		list.push_back(new Employee("Jane"));
		list.push_back(new Manager("Ted"));
	}
	void addEmployee(Employee *p) {
		list.push_back(p);
		getStreamer() << "New employee added.   Current employees are:" << "\n";
		std::vector<Employee*>::iterator i;
		for (i=list.begin(); i!=list.end(); i++) {
			getStreamer() << "  " << (*i)->getTitle() << "\n";
		}
	}
	const Employee *get_item(int i) {
		return list[i];
	}
	~EmployeeList() { 
		std::vector<Employee*>::iterator i;
		getStreamer() << "~EmployeeList, deleting " << list.size() << " employees." << "\n";
		for (i=list.begin(); i!=list.end(); i++) {
			delete *i;
		}
		getStreamer() << "~EmployeeList empty." << "\n";
	}
};

