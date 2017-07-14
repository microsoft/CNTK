CPP = g++
CFLAGS := -Wall -O3 -std=c++11

OBJS = pyreallocate.o
LIB = LightRNN/libpyreallocate.so
DIR_SRC = DLL/DLL/

all : $(LIB)

%.o : ${DIR_SRC}%.cpp
	$(CC) $(CFLAGS) -fpic -c $< -o $@

$(LIB) : $(OBJS)
	rm -f $@
	g++ -shared -o $@ $(OBJS)
	rm -f $(OBJS)

tags:
	ctags -R *

clean:
	rm -f $(OBJS) $(TARGET) $(LIB)
