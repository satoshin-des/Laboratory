CC	= g++
SRC	= BKZ.cpp
CFLAGS	= -shared -fPIC -O3 -fopenmp -mtune=native -march=native -mfpmath=both -unroll-loops -o libbkz.so
LDFLAGS = -lntl

all:
	${CC} ${CFLAGS} ${SRC} ${LDFLAGS}
