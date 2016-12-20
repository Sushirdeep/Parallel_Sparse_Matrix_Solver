IN_DIR =../include
CC=gcc -fopenmp
CFLAGS=-I$(IN_DIR) -std=c99 -lm -O3
OUT_DIR=obj

_DEPS = mmio.h
DEPS = $(patsubst %,$(IN_DIR)/%,$(_DEPS))

_OBJ = mmio.o ConjugateGradient.o main.o  
OBJ = $(patsubst %,$(OUT_DIR)/%,$(_OBJ))


$(OUT_DIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

../main: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f $(OUT_DIR)/*.o *~ core $(INCDIR)/*~ 


