/* -*- mode: c++ -*- */
/* File : example.h -- Tests all string typemaps */

#include <sys/time.h>
#include <time.h>

void takes_std_string( std::string in ) {
    cout << "takes_std_string( \"" << in << "\" );" << endl;
}

std::string gives_std_string() {
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return std::string( asctime( localtime( &tv.tv_sec ) ) );
}

void takes_char_ptr( char *p ) {
    cout << "takes_char_ptr( \"" << p << "\" );" << endl;
}

char *gives_char_ptr() {
    return "foo";
}

void takes_and_gives_std_string( std::string &inout ) {
    inout.insert( inout.begin(), '[' );
    inout.insert( inout.end(), ']' );
}

void takes_and_gives_char_ptr( char *&inout ) {
    char *pout = strchr( inout, '.' );
    if( pout ) inout = pout + 1;
    else inout = "foo";
}

/*
 * Local-Variables:
 * c-indentation-style: "stroustrup"
 * End:
 */
