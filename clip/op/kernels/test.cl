

kernel void test(global double * data, int k)
{
		int i = get_global_id(0);
		printf("x[%d] = %f, k = %d\n", i, data[i], k);
}

kernel void test2 (global double * x)
{
		x[0] = x[0] + 1;
}