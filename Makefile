CC = gcc
LD = $(CC)

CFLAGS = -std=gnu11 -Wall -Wextra -Werror -pedantic -pipe -march=native -g -fopenmp -static
OFLAGS =
LFLAGS = -lm -lc -lOpenCL -lpthread -fopenmp

OPTIMIZATION = -Ofast

CFLAGS += $(OPTIMIZATION)

OBJECTS = cl_common.o benchmark.o cpu_bench.o

all: clperf

clperf: clperf.o $(OBJECTS)
	$(LD) $< $(OBJECTS) $(LFLAGS) -o clperf

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf clperf gmon.out *.save *.o core* vgcore*

rebuild: clean all

.PHONY : clean
.SILENT : clean
.NOTPARALLEL : clean
