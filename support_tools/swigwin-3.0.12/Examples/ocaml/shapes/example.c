/* File : example.c */
#include <stdio.h>
#include "example.h"

shape::~shape() { }

bool shape::cover( double x, double y ) { return false; }

void draw_shape_coverage( shape *s, int div_x, int div_y ) {
    double i,j;

    for( i = 0; i < 1.0; i += 1.0 / ((float)div_y) ) {
	for( j = 0; j < 1.0; j += 1.0 / ((float)div_x) ) {
	    if( s->cover( j,i ) ) putchar( 'x' ); else putchar( ' ' );
	}
	printf( "\n" );
    }
}

void draw_depth_map( volume *v, int div_x, int div_y ) {
    double i,j;
    char depth_map_chars[] = "#*+o;:,. ";
    double lowbound, highbound;
    double current = 0.0;
    bool bounds_set = false;

    for( i = 0; i < 1.0; i += 1.0 / ((float)div_y) ) {
	for( j = 0; j < 1.0; j += 1.0 / ((float)div_x) ) {
	    current = v->depth( j,i );
	    if( !bounds_set ) { 
		lowbound = current; highbound = current; bounds_set = true;
	    }
	    if( current < lowbound ) lowbound = current;
	    if( current > highbound ) highbound = current;
	}
    }

    for( i = 0; i < 1.0; i += 1.0 / ((float)div_y) ) {
	for( j = 0; j < 1.0; j += 1.0 / ((float)div_x) ) {
	    current = ((v->depth( j,i ) - lowbound) / 
		       (highbound - lowbound)) * 8;
	    putchar(depth_map_chars[(int)current]);
	}
	putchar('\n');
    }
}

double volume::depth( double x, double y ) { return 0.0; }

volume::~volume() { }
