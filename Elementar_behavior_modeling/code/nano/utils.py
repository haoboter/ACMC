"""Shared utility functions for active learning scripts.

Provides common helpers for data smoothing and monitoring visualization.
"""
import matplotlib.pyplot as plt
import numpy as np


def moving_average(data, window_size):
    """Compute moving average with specified window size.
    
    Args:
        data: List or array of numeric values.
        window_size: Number of elements in the sliding window.
        
    Returns:
        List of averaged values with same length as input.
    """
    averages = []
    for i in range(len(data)):
        start_index = max(0, i - window_size + 1)
        current_window = data[start_index:i + 1]
        averages.append(sum(current_window) / len(current_window))
    return averages


def plot_monitoring_curves(steps, min_confidences, avg_confidences, errors, error_rates, 
                           error_scale=1.0, error_label_suffix=''):
    """Generate monitoring plots: min confidence, avg confidence, errors, error rates, and combined.
    
    Args:
        steps: List of iteration indices.
        min_confidences: Minimum confidence values per iteration.
        avg_confidences: Average confidence values per iteration.
        errors: Prediction errors per iteration.
        error_rates: Error rates per iteration.
        error_scale: Multiplier for error values in combined plot.
        error_label_suffix: Additional text for error label.
    """
    errors_avg = moving_average(errors, 100)
    errors_avg_scaled = [value * error_scale for value in errors_avg]
    error_rates_avg = moving_average(error_rates, 100)
    error_rates_avg_percentage = [value * 100 for value in error_rates_avg]
    error_product_avg = moving_average([errors[i] * error_rates[i] for i in range(len(errors))], 100)

    # Plot 1: Minimal confidence
    plt.figure(figsize=(10, 6))
    plt.plot(steps, min_confidences, marker='o', linestyle='-', color='b', label='Min Confidence')
    plt.title('Minimal Confidence Value Over Iterations')
    plt.xlabel('Steps')
    plt.ylabel('Minimal Confidence Value')
    plt.xlim(0, np.max(steps) + 1)
    plt.ylim(1.2 * np.min(min_confidences), 1.2 * np.max(min_confidences))
    plt.grid()
    plt.legend()
    plt.savefig('confidence_min_plot.png')

    # Plot 2: Errors (moving average)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, errors_avg, marker='o', linestyle='-', color='r', label='Errors (Moving Avg)')
    plt.title('Errors (Moving Average) Over Iterations')
    plt.xlabel('Steps')
    plt.ylabel('Errors')
    plt.xlim(0, max(steps) + 1)
    plt.ylim(1.2 * min(errors_avg), 1.2 * max(errors_avg))
    plt.grid()
    plt.legend()
    plt.savefig('errors_avg_plot.png')

    # Plot 3: Error rates (moving average)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, error_rates_avg, marker='o', linestyle='-', color='g', label='Error Rates (Moving Avg)')
    plt.title('Error Rates (Moving Average) Over Iterations')
    plt.xlabel('Steps')
    plt.ylabel('Error Rates')
    plt.xlim(0, max(steps) + 1)
    plt.ylim(1.2 * min(error_rates_avg), 1.2 * max(error_rates_avg))
    plt.grid()
    plt.legend()
    plt.savefig('error_rates_avg_plot.png')

    # Plot 4: Error product (moving average)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, error_product_avg, marker='o', linestyle='-', color='m',
             label='Errors * Error Rates (Moving Avg)')
    plt.title('Product of Errors and Error Rates (Moving Average) Over Iterations')
    plt.xlabel('Steps')
    plt.ylabel('Product of Errors and Error Rates')
    plt.xlim(0, max(steps) + 1)
    plt.ylim(1.2 * min(error_product_avg), 1.2 * max(error_product_avg))
    plt.grid()
    plt.legend()
    plt.savefig('error_product_avg_plot.png')

    # Plot 5: Combined plot with dual y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(steps[1:], avg_confidences[1:], marker='o', linestyle='-', color='b', label='Avg Confidence')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Minimum Confidence Value', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xlim(0, np.max(steps) + 1)
    ax1.set_ylim(1.2 * np.min(avg_confidences), 1.2 * np.max(avg_confidences))

    ax2 = ax1.twinx()
    error_label = f'Errors{error_label_suffix} (Moving Avg)' if error_label_suffix else 'Errors (Moving Avg)'
    ax2.plot(steps[1:], errors_avg_scaled[1:], marker='o', linestyle='-', color='r', label=error_label)
    ax2.plot(steps[1:], error_rates_avg_percentage[1:], marker='o', linestyle='-', color='g',
             label='Error Rates (Moving Avg)')
    ax2.plot(steps[1:], error_product_avg[1:], marker='o', linestyle='-', color='m',
             label='Errors * Error Rates (Moving Avg)')
    ax2.set_ylabel('Average Errors / Error Rates', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)
    plt.title('Confidence, Average Errors, and Average Error Rates Over Iterations')
    plt.grid()
    plt.savefig('combined_plot.png')

    plt.close('all')
