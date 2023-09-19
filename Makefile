build: main.zig
	zig build-exe $?

clean: main main.o
	rm $?

.PHONY: clean build