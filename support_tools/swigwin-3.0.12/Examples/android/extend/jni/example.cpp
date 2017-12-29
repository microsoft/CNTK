/* File : example.cxx */

#include "example.h"

static Streamer * streamerInstance = 0;

void setStreamer(Streamer* streamer) {
  streamerInstance = streamer;
}

Streamer& getStreamer() {
  return *streamerInstance;
}

