//__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void vectorAdd(__global float* s, __constant const float* l, __local float* q)
{
	const int gid = get_global_id(0);
	const int lid = get_local_id(0);
	q[lid] = l[lid];
//	for (int i = 0; i < 3e+2; ++i)
//	for (int j = 0; j < 2e+2; ++j)
//	s[gid] = q[lid] * 2.0f + 1.0f;
	s[gid] = s[gid] + q[lid] * 2.0f + 1.0f;
}
