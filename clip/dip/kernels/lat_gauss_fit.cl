kernel void cl_test()
{
    int g_id_0 = get_global_id(0);
    int g_id_1 = get_global_id(1);
    int g_id_2 = get_global_id(2);
    printf(">> global id: (%d, %d, %d)\n", g_id_0, g_id_1, g_id_2);
}
