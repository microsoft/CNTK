/* File : example.h */

/* This is some kind of engine of some kind 
we will give it some dummy methods for Lua to call*/

class Engine
{
public:
    void start();
    void stop();
    void accelerate(float f);
    void decelerate(float f);
};


/* We also want to pass some events to Lua, so let's have a few classes
to do this.
*/
class Event
{
public:
    enum {STARTUP,KEYPRESS,MOUSEPRESS,SHUTDOWN} mType;
    // etc
};
