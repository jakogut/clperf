CC = gcc
LD = $(CC)

CFLAGS = -std=gnu99 -Wall -pipe -march=native -g
OFLAGS =
LFLAGS = -lm -lc -lOpenCL -lpthread

OPTIMIZATION = -Ofast

CFLAGS += $(OPTIMIZATION)

all: clperf

clperf: clperf.o file.o
	$(LD) $< file.o $(LFLAGS) -o clperf

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf clperf gmon.out *.save *.o core* vgcore*

rebuild: clean all

.PHONY : clean
.SILENT : clean
