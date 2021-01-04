"""
Module that helps plotting and analyzing the results of the simulation runs
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import plotly.graph_objects as go

import fledge.problems
import fledge.plots
import fledge.config

logger = fledge.config.get_logger(__name__)


class Plots(object):
    results_path: str

    def __init__(
            self,
            results_path: str = None
    ):
        self.results_path = results_path

    # [ ] branch power magnitude,
    # [ ] node voltage magnitude
    # [ ] total losses
    # [ ] der dispatch
    # [ ] objective result / system costs
    # [ ] prices over time
    # [ ] branch power magnitude over time per branch
    # [ ] node voltage magnitude over time per node
    # [ ] der output over time
    # [ ] dlmp validation
    # TODO: [ ] load flow calculation comparison --> ask Sebastian

    def save_or_show_plot(
            self,
            plot_name: str,
            timestep: pd.Timestamp = None
    ):
        if self.results_path is None:
            plt.show()
        else:
            if timestep is None:
                file_name = f'{plot_name}.png'
            else:
                file_name = f'{plot_name}_{timestep.strftime("%Y-%m-%d_%H-%M-%S")}.png'
            file_path = os.path.join(self.results_path, file_name)
            plt.savefig(file_path)
            print(f'Saved plot: {file_name}', end="\r")
        plt.close()


class AnalysisManager(object):
    results_path: str
    results_dict: dict
    plotter: Plots

    def __init__(
            self,
            results_dict: dict,
            results_path: str = None
    ):
        self.results_path = results_path
        self.results_dict = results_dict
        self.plotter = Plots(self.results_path)

    def generate_result_plots(
            self
    ):
        logger.info('Called function generate_result_plots but not implemented yet')
        pass
