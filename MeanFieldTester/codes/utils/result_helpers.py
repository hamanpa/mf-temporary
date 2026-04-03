

def compare_mf_snn_results(net_results, 
                           mf_results_list:list, 
                           mf_names_list:list, 
                           start_time:float,
                           values_list:list[str]=None) -> list[dict]:
    """Compares a list of mean-field results with SNN, returns a dictionary of errors.
    
    Comparison is done by following formula:
        error = ((SNN_result - MF_result)**2).mean()

    Parameters
    ----------
    net_results : NetworkResults
        results from the spiking network simulation
    mf_results_list : list of Results
        list of mean-field results to compare with SNN
    mf_names_list : list of str
        list of names for the mean-field results
    start_time : float
        time [ms] from which to start averaging the results
    values_list : list of str, optional
        list of values to compare, by default None. If None, compares:
            "exc_rate_mean", 
            "inh_rate_mean", 
            "exc_adaptation_mean", 
            "exc_voltage_mean", 
            "inh_voltage_mean"
    
    Returns
    -------
    errors : list of dict
        list of dictionaries containing the errors for each mean-field result

    """

    if values_list is None:
        values = [
            "exc_rate_mean", 
            "inh_rate_mean", 
            "exc_adaptation_mean", 
            "exc_voltage_mean", 
            "inh_voltage_mean"
        ]
    
    if mf_names_list is None:
        mf_names_list = [f"MF{i}" for i in range(len(mf_results_list))]

    errors = []

    for mf_results, mf_name in zip(mf_results_list, mf_names_list):
        errors.append(compare_single_mf_snn_results(net_results, mf_results, start_time, values_list))

    return errors


def compare_single_mf_snn_results(net_results, mf_results, start_time:float, values_list:list[str]=None) -> dict:
    """Compares a list of mean-field results with spiking network simulation, returns a dictionary of errors.

    Comparison is done by following formula:
        error = ((SNN_result - MF_result)**2).mean()

    Parameters
    ----------
    net_results : NetworkResults
        results from the spiking network simulation
    mf_results_list : list of Results
        list of mean-field results to compare with spiking network
    start_time : float
        time [ms] from which to start averaging the results
    values_list : list of str, optional
        list of values to compare, by default None. If None, compares:
            "exc_rate_mean", 
            "inh_rate_mean", 
            "exc_adaptation_mean", 
            "exc_voltage_mean", 
            "inh_voltage_mean"
    
    Returns
    -------
    errors : list of dict
        list of dictionaries containing the errors for each mean-field result

    """



    # Set smoothing function for SNN results

    errors = dict()

    if values_list is None:
        values_list = [
            "exc_rate_mean", 
            "inh_rate_mean", 
            "exc_adaptation_mean", 
            "exc_voltage_mean", 
            "inh_voltage_mean"
        ]

    for value in values_list:
        snn_data = getattr(net_results, value)[net_results.times >= start_time]
        mf_data = getattr(mf_results, value)[mf_results.times >= start_time]
        error = ((snn_data - mf_data)**2).mean()
        errors[value] = error

    return errors