//__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void monte_carlo(__global float* const s, __constant const float* const l, __local float* const q, __constant const float const p[16])
{
	const int gid = get_global_id(0);
	const int lid = get_local_id(0);
	q[lid] = l[lid];
	for (int j = 0; j < 2e+3; ++j)
	for (int i = 0; i < 1e+3; ++i)
	s[gid] = s[gid] + q[lid] * 2.0f + 1.0f + p[lid % 16];
	s[gid] = 0;
	s[gid] = s[gid] + q[lid] * 2.0f + 1.0f + p[lid % 16];
}
