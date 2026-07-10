
from ..data_structures.base import BaseInspectionResults
from ..data_structures.inspection import ComparisonInspectionResults, SteadyStateInspectionResults
from .base import BaseInspectionPlot
from typing import List


class CustomInspectionPlot(BaseInspectionPlot):

    def _draw(self, ax, inspection_results_list:List[BaseInspectionResults], variable=None):

        inspection_results = self.filter_results(inspection_results_list)[0]

        if variable not in inspection_results.measured_variables:
            return  # Skip plotting if the variable is not present in the inspection results

        self.update_params(inspection_results)
        param_values = inspection_results.param_values

        for measured_values, ls, marker, label in zip(getattr(inspection_results, variable)(), self.full_params['linestyles'], self.full_params['markers'], self.full_params['labels']):
            ax.plot(
                param_values, 
                measured_values, 
                label=label,
                linestyle=ls,
                marker=marker,
                color="black",
            )

class MetricCustomInspectionPlot(CustomInspectionPlot):

    def __init__(self, metric_name: str, plot_params: dict):
        self.metric_name = metric_name
        super().__init__(plot_params)


    def _draw(self, ax, inspection_results: BaseInspectionResults):
        super()._draw(ax, inspection_results, variable=self.metric_name)




class FiringRateInspectionPlot(BaseInspectionPlot):
    """Plot the firing rate of excitatory and inhibitory neurons over time."""
    DEFAULT_PARAMS = {
        **BaseInspectionPlot.DEFAULT_PARAMS,
        'title': 'Time averaged Firing Rate\n vs Parameter',
        'xlabel': None,  # Will be set to inspected_param
        'ylabel': 'Firing Rate (Hz)',
    }

    inspection_results_type = SteadyStateInspectionResults 

    def _draw(self, ax, inspection_results_list:List[BaseInspectionResults]):
        inspection_results = self.filter_results(inspection_results_list)[0]

        self.update_params(inspection_results)
        param_values = inspection_results.param_values
        
        has_exc_mean = "exc_rate_time_mean" in inspection_results.measured_variables
        has_exc_std = "exc_rate_time_std" in inspection_results.measured_variables
        has_inh_mean = "inh_rate_time_mean" in inspection_results.measured_variables
        has_inh_std = "inh_rate_time_std" in inspection_results.measured_variables
        
        exc_mean_data = inspection_results.exc_rate_time_mean() if has_exc_mean else None
        exc_std_data = inspection_results.exc_rate_time_std() if has_exc_std else None
        inh_mean_data = inspection_results.inh_rate_time_mean() if has_inh_mean else None
        inh_std_data = inspection_results.inh_rate_time_std() if has_inh_std else None

        for i, (network_name, ls, marker, label) in enumerate(zip(inspection_results.network_names, self.full_params['linestyles'], self.full_params['markers'], self.full_params['labels'])):
            print("WARNING: spiking network has to be named 'SNN' inside inspection_results.network_names to work properly!")
            is_snn = network_name.startswith("SNN")
            
            if has_exc_mean:
                if is_snn and has_exc_std:
                    ax.errorbar(param_values, exc_mean_data[i], yerr=exc_std_data[i], label=f'Exc {label}', ls=ls, marker=marker, color=self.full_params['exc_color'])
                else:
                    ax.plot(param_values, exc_mean_data[i], label=f'Exc {label}', ls=ls, marker=marker, color=self.full_params['exc_color'])
                    if not is_snn and has_exc_std:
                        ax.fill_between(param_values, exc_mean_data[i] - exc_std_data[i], exc_mean_data[i] + exc_std_data[i], color=self.full_params['exc_color'], alpha=0.3)
            
            if has_inh_mean:
                if is_snn and has_inh_std:
                    ax.errorbar(param_values, inh_mean_data[i], yerr=inh_std_data[i], label=f'Inh {label}', ls=ls, marker=marker, color=self.full_params['inh_color'])
                else:
                    ax.plot(param_values, inh_mean_data[i], label=f'Inh {label}', ls=ls, marker=marker, color=self.full_params['inh_color'])
                    if not is_snn and has_inh_std:
                        ax.fill_between(param_values, inh_mean_data[i] - inh_std_data[i], inh_mean_data[i] + inh_std_data[i], color=self.full_params['inh_color'], alpha=0.3)


class VoltageInspectionPlot(BaseInspectionPlot):
    """Plot the firing rate of excitatory and inhibitory neurons over time."""
    DEFAULT_PARAMS = {
        **BaseInspectionPlot.DEFAULT_PARAMS,
        'title': 'Time averaged Voltage\n vs Parameter',
        'xlabel': None,  # Will be set to inspected_param
        'ylabel': 'Voltage (mV)',
    }

    inspection_results_type = SteadyStateInspectionResults

    def _draw(self, ax, inspection_results_list:List[BaseInspectionResults]):
        inspection_results = self.filter_results(inspection_results_list)[0]

        self.update_params(inspection_results)
        param_values = inspection_results.param_values

        has_exc_mean = "exc_voltage_time_mean" in inspection_results.measured_variables
        has_exc_std = "exc_voltage_time_std" in inspection_results.measured_variables
        has_inh_mean = "inh_voltage_time_mean" in inspection_results.measured_variables
        has_inh_std = "inh_voltage_time_std" in inspection_results.measured_variables

        exc_mean_data = inspection_results.exc_voltage_time_mean() if has_exc_mean else None
        exc_std_data = inspection_results.exc_voltage_time_std() if has_exc_std else None
        inh_mean_data = inspection_results.inh_voltage_time_mean() if has_inh_mean else None
        inh_std_data = inspection_results.inh_voltage_time_std() if has_inh_std else None

        for i, (network_name, ls, marker, label) in enumerate(zip(inspection_results.network_names, self.full_params['linestyles'], self.full_params['markers'], self.full_params['labels'])):
            print("WARNING: spiking network has to be named 'SNN' inside inspection_results.network_names to work properly!")
            is_snn = network_name.startswith("SNN")

            if has_exc_mean:
                if is_snn and has_exc_std:
                    ax.errorbar(param_values, exc_mean_data[i], yerr=exc_std_data[i], label=f'Exc {label}', ls=ls, marker=marker, color=self.full_params['exc_color'])
                else:
                    ax.plot(param_values, exc_mean_data[i], label=f'Exc {label}', ls=ls, marker=marker, color=self.full_params['exc_color'])
            
            if has_inh_mean:
                if is_snn and has_inh_std:
                    ax.errorbar(param_values, inh_mean_data[i], yerr=inh_std_data[i], label=f'Inh {label}', ls=ls, marker=marker, color=self.full_params['inh_color'])
                else:
                    ax.plot(param_values, inh_mean_data[i], label=f'Inh {label}', ls=ls, marker=marker, color=self.full_params['inh_color'])

class AdaptationInspectionPlot(BaseInspectionPlot):
    """Plot the firing rate of excitatory and inhibitory neurons over time."""
    DEFAULT_PARAMS = {
        **BaseInspectionPlot.DEFAULT_PARAMS,
        'title': 'Time averaged Adaptation\n vs Parameter',
        'xlabel': None,  # Will be set to inspected_param
        'ylabel': 'Adaptation (pA)',
    }

    inspection_results_type = SteadyStateInspectionResults 

    def _draw(self, ax, inspection_results_list:List[BaseInspectionResults]):
        inspection_results = self.filter_results(inspection_results_list)[0]

        self.update_params(inspection_results)
        param_values = inspection_results.param_values

        has_exc_mean = "exc_adaptation_time_mean" in inspection_results.measured_variables
        has_exc_std = "exc_adaptation_time_std" in inspection_results.measured_variables

        exc_mean_data = inspection_results.exc_adaptation_time_mean() if has_exc_mean else None
        exc_std_data = inspection_results.exc_adaptation_time_std() if has_exc_std else None

        for i, (network_name, ls, marker, label) in enumerate(zip(inspection_results.network_names, self.full_params['linestyles'], self.full_params['markers'], self.full_params['labels'])):
            print("WARNING: spiking network has to be named 'SNN' inside inspection_results.network_names to work properly!")
            is_snn = network_name.startswith("SNN")

            if has_exc_mean:
                if is_snn and has_exc_std:
                    ax.errorbar(param_values, exc_mean_data[i], yerr=exc_std_data[i], label=f'Exc {label}', ls=ls, marker=marker, color='blue')
                else:
                    ax.plot(param_values, exc_mean_data[i], label=f'Exc {label}', ls=ls, color='blue', marker=marker)