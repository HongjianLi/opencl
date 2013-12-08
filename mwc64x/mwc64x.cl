/*
Copyright (c) 2011, David Thomas
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

	* Redistributions of source code must retain the above copyright notice,
	this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	notice, this list of conditions and the following disclaimer in the
	documentation and/or other materials provided with the distribution.
	* Neither the name of Imperial College London nor the names of its
	contributors may be used to endorse or promote products derived
	from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Pre: a < M, b < M
// Post: r = (a + b) mod M
ulong AddMod64(ulong a, ulong b, ulong M)
{
	ulong v = a + b;
	if (v >= M || v < a) v -= M;
	return v;
}

// Pre: a < M, b < M
// Post: r = (a * b) mod M
ulong MulMod64(ulong a, ulong b, ulong M)
{	
	ulong r = 0;
	while (a)
	{
		if (a & 1) r = AddMod64(r, b, M);
		b = AddMod64(b, b, M);
		a = a >> 1;
	}
	return r;
}

// Pre: a < M, e >= 0
// Post: r = (a ^ b) mod M
// This takes at most ~64^2 modular additions, so probably about 2^15 or so instructions on
// most architectures
ulong PowMod64(ulong a, ulong e, ulong M)
{
	ulong sqr = a, acc = 1;
	while (e)
	{
		if (e & 1) acc = MulMod64(acc, sqr, M);
		sqr = MulMod64(sqr, sqr, M);
		e = e >> 1;
	}
	return acc;
}

typedef struct { uint x; uint c; } mwc64x_state_t;

enum { A = 4294883355U };
enum { M = 18446383549859758079UL };
enum { B = 4077358422479273989UL };

void skip(mwc64x_state_t *s, ulong d)
{
	ulong m = PowMod64(A, d, M);
	ulong x = MulMod64(s->x * (ulong)A + s->c, m, M);
	s->x = x / A;
	s->c = x % A;
}

void seed(mwc64x_state_t *s, ulong baseOffset, ulong perStreamOffset)
{
	ulong d = baseOffset + get_global_id(0) * perStreamOffset;
	ulong m = PowMod64(A, d, M);
	ulong x = MulMod64(B, m, M);
	s->x = x / A;
	s->c = x % A;
}

uint next(mwc64x_state_t *s)
{
	uint X = s->x;
	uint C = s->c;
	uint r = X ^ C;
	uint Xn = A * X + C;
	uint carry = Xn < C;
	uint Cn = mad_hi(A, X, carry);
	s->x = Xn;
	s->c = Cn;
	return r;
}

__kernel void EstimatePi(ulong n, ulong baseOffset, __global ulong *acc)
{
	mwc64x_state_t rng;
	ulong samplesPerStream = n / get_global_size(0);
	seed(&rng, baseOffset, 2 * samplesPerStream);
	uint count = 0;
	for (uint i = 0; i < samplesPerStream; ++i)
	{
		ulong x = next(&rng);
		ulong y = next(&rng);
		ulong x2 = x * x;
		ulong y2 = y * y;
		if (x2+y2 >= x2) ++count;
	}
	acc[get_global_id(0)] = count;
}
