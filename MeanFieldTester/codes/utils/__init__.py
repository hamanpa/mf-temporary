


# TODO: move this to a corresponding helpers file
def compare_mf_snn_results(net_results, mf_results_list:list, mf_names_list, start_time:float):
    if mf_names_list is None:
        mf_names_list = [f"MF{i}" for i in range(len(mf_results_list))]

    net_results.print_time_averaged(start_time=start_time)
    for mf_results, mf_name in zip(mf_results_list, mf_names_list):
        print(f"Results for {mf_name}:")
        mf_results.print_time_averaged(start_time=start_time)
