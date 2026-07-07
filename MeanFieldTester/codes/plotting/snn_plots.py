import numpy as np

from ..utils.snn_helpers import activity_from_spikes_histogram

from ..data_structures.base import BaseSNNResults
from .base import BaseSNNPlot

from .base import RASTER_EXC_CELLS, RASTER_INH_CELLS, BIN_SIZE


class SpikeRasterPlot(BaseSNNPlot):
    """Plot the spike raster of excitatory and inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseSNNPlot.DEFAULT_PARAMS,
        'title': 'Spike Raster Plot',
        'xlabel': 'Time (ms)',
        'ylabel': 'Neuron Index',
        'marker': 'o',
        'markersize': 7,
        'exc_cells': RASTER_EXC_CELLS,
        'inh_cells': RASTER_INH_CELLS,
    }

    def _draw(self, ax, snn_results:BaseSNNResults):
        exc_cells = self.full_params['exc_cells']
        inh_cells = self.full_params['inh_cells']
        for i, spiketrain in enumerate(snn_results.exc_spikes_all()[:exc_cells], start=1):
            ax.scatter(spiketrain, i * np.ones_like(spiketrain), color=self.full_params['exc_color'], 
                       marker=self.full_params['marker'], s=self.full_params['markersize'], lw=0)
        for i, spiketrain in enumerate(snn_results.inh_spikes_all()[:inh_cells], start=exc_cells + 1):
            ax.scatter(spiketrain, i * np.ones_like(spiketrain), color=self.full_params['inh_color'],
                       marker=self.full_params['marker'], s=self.full_params['markersize'], lw=0)


class ActivityHistogramPlot(BaseSNNPlot):
    """Plot the activity histogram of excitatory and inhibitory neurons."""
    DEFAULT_PARAMS = {
        **BaseSNNPlot.DEFAULT_PARAMS,
        'title': 'Activity Histogram',
        'xlabel': 'Time (ms)',
        'ylabel': 'Firing Rate (Hz)',
        'binsize': BIN_SIZE,  # Size of the bins for the histogram
    }

    def _draw(self, ax, snn_results:BaseSNNResults):
        exc_activity = activity_from_spikes_histogram(snn_results.exc_spikes_all(), snn_results.times(), self.full_params['binsize'])
        inh_activity = activity_from_spikes_histogram(snn_results.inh_spikes_all(), snn_results.times(), self.full_params['binsize'])
        
        ax.plot(snn_results.times(), exc_activity, label='Excitatory', color=self.full_params['exc_color'])
        ax.plot(snn_results.times(), inh_activity, label='Inhibitory', color=self.full_params['inh_color'])

