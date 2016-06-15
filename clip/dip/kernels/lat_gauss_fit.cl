kernel void cl_test(int x)
{
    int g_id_0 = get_global_id(0);
    int g_id_1 = get_global_id(1);
    
    printf(">> global id: (%d, %d), x = %d\n", g_id_0, g_id_1, x);
}

kernel void sub_ado(global float *data, float ado)
{
		size_t i = get_global_id(0);  			
		size_t j = get_global_id(1);  			
		
		size_t m = get_global_size(0);
		
		data[j * m + i] -= ado;
}

