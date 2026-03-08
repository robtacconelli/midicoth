CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra
LDFLAGS = -lm

HEADERS = arith.h ppm.h tweedie.h match.h word.h highctx.h fastmath.h

all: mdc ablation

mdc: mdc.c $(HEADERS)
	$(CC) $(CFLAGS) -o mdc mdc.c $(LDFLAGS)

ablation: ablation.c $(HEADERS)
	$(CC) $(CFLAGS) -o ablation ablation.c $(LDFLAGS)

test_arith: test_arith.c arith.h
	$(CC) $(CFLAGS) -o test_arith test_arith.c $(LDFLAGS)

test_ppm: test_ppm.c arith.h ppm.h
	$(CC) $(CFLAGS) -o test_ppm test_ppm.c $(LDFLAGS)

clean:
	rm -f mdc ablation bench test_arith test_ppm

.PHONY: all clean
