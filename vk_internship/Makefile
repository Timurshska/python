# Makefile
.PHONY: all clean

all: features submission clean
features:
	python feature_engineering.py
submission: features
	python generate_submission.py
clean:
ifeq ($(OS),Windows_NT)
	del /Q *.npy
else
	rm -f *.npy
endif
