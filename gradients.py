from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt, float, amax


def read_format_data(filename):
    points = genfromtxt(filename + ".csv", delimiter=",", dtype=float, names=["Time", "Strain"])
    # Time = x (no manipulation), Strain = y (divide by initial length)

    initial_length = {
        "Sample_1": 45.98,
        "Sample_2": 67.04,
        "Sample_3": 44.33,
    }

    points["Strain"] = [x / float(initial_length[filename]) for x in points["Strain"]]

    return points


def regression(points, start, end):  # add error value calculation instead of stderr
    (m_value, b_value, r_value, tt, stderr) = stats.linregress(points["Time"][start:end],
                                                               points["Strain"][start:end])

    err = error_estimate(points, m_value)
    print('regression: a=%.4f b=%.4f, r=%.2f' % (m_value, b_value, r_value))
    return [m_value, b_value, r_value, err]


def error_estimate(points, m_value):
    y_bar = np.mean(points["Strain"])
    x_bar = np.mean(points["Time"])

    sq_dev_y = 0
    sq_dev_x = 0
    for i in points["Strain"]:
        sq_dev_y += (i - y_bar) ** 2
    for i in points["Time"]:
        sq_dev_x += (i - x_bar) ** 2

    delta_e = np.abs((sq_dev_y / sq_dev_x) - (m_value ** 2))
    err = np.sqrt(delta_e / (points["Time"].size - 2))

    return err


def f(x, m, b):
    return m * x + b


def plot_values(names, points_list, reg_values_list, start, end):
    color_list = {
        "Sample_1": "ko",
        "Sample_2": "bo",
        "Sample_3": "ro",
    }

    start_d = {}
    end_d = {}
    x_max = 0
    y_max = 0
    for i in range(3):
        start_d["Sample_" + str(i + 1)] = start[i]
        end_d["Sample_" + str(i + 1)] = end[i]

        if amax(points_list[i]["Time"]) > x_max:
            x_max = amax(points_list[i]["Time"])
        if amax(points_list[i]["Strain"]) > y_max:
            y_max = amax(points_list[i]["Strain"])

    spacing_x = x_max + x_max / 5
    spacing_y = y_max + y_max / 5

    plt.xlabel("Time (s {} 0.001)".format(u"\u00B1"))
    plt.ylabel("Strain (% {} 0.02)".format(u"\u00B1"))

    plt.axis([0, spacing_x, 0, spacing_y])

    for i, name, points, reg_values in zip(range(3), names, points_list, reg_values_list):
        plt.plot(points["Time"], points["Strain"], color_list[name], label=name.replace("_", " "))

        plt.text(spacing_x * 0.05, spacing_y * (0.9 - i * 0.1),
                 "Linear Fit Equation for {4}: \ny ={0:.4f}x {5} {3:.4f}+ {1:.3f} %/s \nr = {2:.2f}".format(
                     reg_values[0],
                     reg_values[1],
                     reg_values[2],
                     reg_values[3],
                     name.replace("_", " "),
                     u"\u00B1"))

        plt.plot(points["Time"][start_d[name]:end_d[name]],
                 f(points["Time"][start_d[name]:end_d[name]], reg_values[0], reg_values[1]),
                 color_list[name].replace("o", "--"), label="Linear Fit for " + name.replace("_", " "))

        plt.axvline(points["Time"][start_d[name]], ymin=0, ymax=0.65, linewidth=2, color=color_list[name][0])
        plt.axvline(points["Time"][end_d[name] - 1], ymin=0, ymax=0.65, linewidth=2, color=color_list[name][0])

        plt.legend(loc=4, borderaxespad=0.)

    plt.show()


def create_graph(name, start, end):
    points = []
    reg_values = []
    for i in range(3):
        points.append(read_format_data(name[i]))
        reg_values.append(regression(points[i], start[i], end[i]))
    plot_values(name, points, reg_values, start, end)


create_graph(["Sample_1", "Sample_2", "Sample_3"], [3, 6, 5], [8, 14, 11])

plt.close("all")
