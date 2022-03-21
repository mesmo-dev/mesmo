import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
pd.options.plotting.backend = "matplotlib"
import mesmo


def main():
    # Settings.
    scenario_name = 'ieee_34node'
    flexible_der_type = ['flexible_generator', 'flexible_load']
    sample_time = '2021-02-22 14:00:00'

    nominal_operation = mesmo.api.run_nominal_operation_problem(scenario_name, store_results=False)
    nominal_branch_power_1 = nominal_operation.branch_power_magnitude_vector_1_per_unit.max()
    nominal_branch_power_2 = nominal_operation.branch_power_magnitude_vector_2_per_unit.max()
    nominal_voltage = nominal_operation.node_voltage_magnitude_vector_per_unit.min()
    nominal_flexible_der_active_power_per_unit = \
        nominal_operation.der_active_power_vector_per_unit[flexible_der_type]

    # High-level API call.
    optimal_operation = mesmo.api.run_optimal_operation_problem(scenario_name, store_results=False)
    optimal_branch_power_1 = optimal_operation.branch_power_magnitude_vector_1_per_unit.max()
    optimal_branch_power_2 = optimal_operation.branch_power_magnitude_vector_2_per_unit.max()
    optimal_voltage = optimal_operation.node_voltage_magnitude_vector_per_unit.min()
    optimal_flexible_der_active_power_per_unit = \
        optimal_operation.der_active_power_vector_per_unit[flexible_der_type]

    x = np.arange(len(optimal_flexible_der_active_power_per_unit.columns))
    width = 0.35
    fig, axes = plt.subplots(1, figsize=(20, 6))
    axes.bar(x + width / 2,
             optimal_flexible_der_active_power_per_unit.loc[sample_time],
             width=width,
             color='b',
             label='optimal')
    axes.bar(x - width / 2,
             nominal_flexible_der_active_power_per_unit.loc[sample_time],
             width=width,
             color='r',
             label='nominal')
    axes.set_xticks(x, optimal_flexible_der_active_power_per_unit.columns)
    plt.xlabel('DER name')
    fig.set_tight_layout(True)
    axes.set_ylabel('Power dispatch [p.u]')
    axes.title.set_text(f"Flexible DER's active power dispatch at {sample_time}")
    plt.xticks(rotation=-90, fontsize=10)
    axes.legend()
    axes.grid()
    fig.show()

    fig, axes = plt.subplots(3, figsize=(12, 12))
    for i in [1, 2, 3]:
        optimal_voltage[:, :, i].plot(
            ax=axes[i - 1],
            label=f'Min optimal voltage profile phase {i}',
            # y=(slice(None), slice(None), 3),
            color='b',
            marker='*'
        )
        nominal_voltage[:, :, i].plot(
            ax=axes[i - 1],
            label=f'Min nominal voltage profile phase {i}',
            # y=(slice(None), slice(None), 3),
            color='r',
            marker='*'
        )
        x = np.arange(len(optimal_voltage[:, :, i].index))
        axes[i - 1].set_xticks(x)
        axes[i - 1].set_xticklabels(optimal_voltage[:, :, i].index, rotation=-30, fontsize=8, minor=False)
        fig.suptitle(f'Nodal Voltage Profile at {sample_time}')
        axes[i - 1].set_ylim([0.5, 1.05])
        axes[i - 1].set_ylabel('Voltage [p.u]')
        axes[i - 1].legend()
        axes[i - 1].grid(axis='y')
        axes[i - 1].grid(axis='x')
        fig.set_tight_layout(True)
        fig.set_tight_layout(True)
    fig.show()

    fig, axes = plt.subplots(3, figsize=(12, 12))
    for i in [1, 2, 3]:
        optimal_branch_power_1.loc['line', slice(None), i].plot(
            ax=axes[i - 1],
            label=f'Max optimal line loading phase {i}',
            color='b',
            marker='*'
        )
        nominal_branch_power_1.loc['line', slice(None), i].plot(
            ax=axes[i - 1],
            label=f'Max nominal line loading phase {i}',
            color='r',
            marker='*'
        )
    for i in [1, 2, 3]:
        x = np.arange(len(nominal_branch_power_1[:, :, i].index[:-2]))
        axes[i - 1].set_xticks(x)
        axes[i - 1].set_xticklabels(nominal_branch_power_1[:, :, i].index[:-2], rotation=-30, fontsize=8, minor=False)
        # axes[i-1].set_ylim([0, 7])
        axes[i - 1].set_ylabel('Loading [p.u]')
        axes[i - 1].legend()
        axes[i - 1].grid(axis='y')
        axes[i - 1].grid(axis='x')
        fig.set_tight_layout(True)
        fig.suptitle('Max Line loading')
        fig.set_tight_layout(True)
    fig.show()

    fig, axes = plt.subplots(3, figsize=(12, 12))
    for i in [1, 2, 3]:
        optimal_branch_power_2.loc['line', slice(None), i].plot(
            ax=axes[i - 1],
            label=f'Max optimal line loading phase {i}',
            color='b',
            marker='*'
        )
        nominal_branch_power_2.loc['line', slice(None), i].plot(
            ax=axes[i - 1],
            label=f'Max nominal line loading phase {i}',
            color='r',
            marker='*'
        )
    for i in [1, 2, 3]:
        x = np.arange(len(nominal_branch_power_2[:, :, i].index[:-2]))
        axes[i - 1].set_xticks(x)
        axes[i - 1].set_xticklabels(nominal_branch_power_2[:, :, i].index[:-2], rotation=-30, fontsize=8, minor=False)
        # axes[i-1].set_ylim([0, 7])
        axes[i - 1].set_ylabel('Loading [p.u]')
        axes[i - 1].legend()
        axes[i - 1].grid(axis='y')
        axes[i - 1].grid(axis='x')
        fig.set_tight_layout(True)
        fig.suptitle('Max Line loading')
        fig.set_tight_layout(True)
    fig.show()


    print(1)


if __name__ == '__main__':
    main()
