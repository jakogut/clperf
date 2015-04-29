#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "file.h"

int clp_fcopy(char *dest, size_t size, FILE* src)
{
	size_t i;
	long pos = ftell(src);

	for(i = 0; i < size && !feof(src); i++) dest[i] = fgetc(src);
	fseek(src, pos, SEEK_SET);

	return 0;
}

int clp_flength(FILE *f)
{
	int length;
	long pos = ftell(f);

	for(length = 0; !feof(f); length++) fgetc(f);
	fseek(f, pos, SEEK_SET);

	return length - 1;
}
