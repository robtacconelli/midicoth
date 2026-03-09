CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra
LDFLAGS = -lm

WINCC = x86_64-w64-mingw32-gcc
WINCFLAGS = -O3 -Wall -Wextra
WINLDFLAGS = -lm -lpthread

HEADERS = arith.h ppm.h tweedie.h match.h word.h highctx.h fastmath.h

all: mdc ablation

mdc: mdc.c $(HEADERS)
	$(CC) $(CFLAGS) -o mdc mdc.c $(LDFLAGS)

ablation: ablation.c $(HEADERS)
	$(CC) $(CFLAGS) -o ablation ablation.c $(LDFLAGS)

win: mdc.exe ablation.exe

mdc.exe: mdc.c $(HEADERS)
	$(WINCC) $(WINCFLAGS) -o mdc.exe mdc.c $(WINLDFLAGS)

ablation.exe: ablation.c $(HEADERS)
	$(WINCC) $(WINCFLAGS) -o ablation.exe ablation.c $(WINLDFLAGS)

test_arith: test_arith.c arith.h
	$(CC) $(CFLAGS) -o test_arith test_arith.c $(LDFLAGS)

test_ppm: test_ppm.c arith.h ppm.h
	$(CC) $(CFLAGS) -o test_ppm test_ppm.c $(LDFLAGS)

clean:
	rm -f mdc ablation bench test_arith test_ppm mdc.exe ablation.exe

.PHONY: all win clean
