/* File : example.h -- stolen from the guile std_vector example */

#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <numeric>

using std::string;

double vec_write(std::vector<string> v) {
    int n = 0;
    for( std::vector<string>::iterator i = v.begin();
	 i != v.end();
	 i++ )
	printf( "%04d: %s\n", ++n, i->c_str() );
}
