%module union_parameter

%warnfilter(SWIGWARN_PARSE_KEYWORD) type; // 'type' is a Go keyword, renamed as 'Xtype'

%inline %{

typedef unsigned char Uint8;

typedef struct SDL_ActiveEvent {
        Uint8 type;     /* SDL_ACTIVEEVENT */
        Uint8 gain;     /* Whether given states were gained or lost (1/0) */
        Uint8 state;    /* A mask of the focus states */
} SDL_ActiveEvent;

/* Keyboard event structure */
typedef struct SDL_KeyboardEvent {
        Uint8 type;     /* SDL_KEYDOWN or SDL_KEYUP */
        int which;    /* The keyboard device index */
        int state;    /* SDL_PRESSED or SDL_RELEASED */
} SDL_KeyboardEvent;

typedef union {
        Uint8 type;
        SDL_ActiveEvent active;
        SDL_KeyboardEvent key;
} SDL_Event;

int SDL_PollEvent (SDL_Event *ev) {
    static int toggle = 0;
    if (toggle == 0) {
        ev->type = 1;
        ev->active.gain = 20;
        ev->active.state = 30;
    } else {
        ev->type = 2;
        ev->key.which = 2000;
        ev->key.state = 3000;
    }
    toggle = 1 - toggle;
    return 1;
}

%}
