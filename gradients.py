from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt, float, amax, amin


def read_format_data(filename, initial_length):
    points = genfromtxt(filename + ".csv", delimiter=",", dtype=float, names=["Time", "Strain"])
    # Time = x (no manipulation), Strain = y (divide by initial length)

    points["Strain"] = [x / float(initial_length[filename]) for x in points["Strain"]]

    return points


def calculate_stress(mass, diameter):
    stress = []
    for m, d in zip(mass.values(), diameter.values()):
        stress.append(((m * 9.81) / (((d * 10 ** -3) / 2) ** 2 * np.pi)))

    return stress


def regression(points, start, end):  # add error value calculation instead of stderr
    (m_value, b_value, r_value, tt, stderr) = stats.linregress(points["Time"][start:end],
                                                               points["Strain"][start:end])

    err = error_estimate(points["Time"], points["Strain"], m_value)
    print('regression: a=%.4f b=%.4f, r=%.2f' % (m_value, b_value, r_value))
    return [m_value, b_value, r_value, err]


def error_estimate(points_x, points_y, m_value):
    x_bar = np.mean(points_x)
    y_bar = np.mean(points_y)

    sq_dev_y = 0
    sq_dev_x = 0
    for i in points_y:
        sq_dev_y += ((i - y_bar) ** 2)

    for i in points_x:
        sq_dev_x += ((i - x_bar) ** 2)

    delta_e = np.abs((sq_dev_y / sq_dev_x) - (m_value ** 2))
    err = np.sqrt(delta_e / (len(points_x) - 2))

    return err


def f(x, m, b):
    return m * x + b


def plot_strain_time(names, points_list, reg_values_list, start, end, color_list):
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

        plt.text(spacing_x * 0.7, spacing_y * (0.370 - i * 0.075),
                 "Linear Fit Equation for {4}: \ny ={0:.4f}x {5} {3:.4f} + {1:.3f} %/s \nr = {2:.2f}".format(
                     reg_values[0],
                     reg_values[1],
                     reg_values[2],
                     reg_values[3],
                     name.replace("_", " "),
                     u"\u00B1"))

        plt.plot(points["Time"][start_d[name]:end_d[name]],
                 f(points["Time"][start_d[name]:end_d[name]], reg_values[0], reg_values[1]),
                 color_list[name].replace("o", "--"), label="Linear Fit for " + name.replace("_", " "))

        plt.axvline(points["Time"][start_d[name]], ymin=0, ymax=0.95, linewidth=2, color=color_list[name][0])
        plt.axvline(points["Time"][end_d[name] - 1], ymin=0, ymax=0.95, linewidth=2, color=color_list[name][0])

        plt.legend(loc=4, borderaxespad=0.)

    plt.show()


def plot_strain_rate_stress(plot_info, stress):
    stress = np.log(stress)
    strain_rate = []
    for i in range(3):
        strain_rate.append(np.log(plot_info[i][0]))

    uncertainty_y = []
    for i in range(3):
        uncertainty_y.append(plot_info[i][3] / plot_info[i][0])

    uncertainty_dia = 0.01 * 10 ** -3
    uncertainty_area_per = uncertainty_dia / (0.9566666 / 2)
    uncertainty_area_per *= 2
    uncertainty_x_per = (0.0001 / 2.3674) + ((0.01 * 10 ** -3) + uncertainty_area_per)
    uncertainty_x = 2.3674 * 9.81 / ((0.9566666 / 2) ** 2 * np.pi) * uncertainty_x_per
    uncertainty_x = uncertainty_x / np.log(2.3674 * 9.81)

    x_max = amax(stress)
    y_max = amin(strain_rate)

    spacing_x = x_max + x_max / 4
    spacing_y = y_max + y_max / 4

    (m_value, b_value, r_value, tt, stderr) = stats.linregress(stress, strain_rate)
    error = np.log(error_estimate(stress, strain_rate, m_value))

    plt.xlabel("Log(Stress) (log(Pa) {0} {1:.3f} )".format(u"\u00B1", uncertainty_x))
    plt.ylabel("Log(Strain Rate)(log(%/s) {0} {1:.3f})".format(u"\u00B1", amax(uncertainty_y)))

    plt.axis([0, spacing_x, spacing_y, 0])

    plt.errorbar(stress, strain_rate, fmt="ko", xerr=0, yerr=uncertainty_y, label="Stress vs Strain-Rate")

    plt.text(spacing_x * 0.05, spacing_y * 0.2,
             "Linear Fit Equation: \ny ={0:.4f}x {4} {3:.4f} + {1:.3f} log(%Pa/s) \nr = {2:.2f}".format(
                 m_value,
                 b_value,
                 r_value,
                 error,
                 u"\u00B1"))
    stress = np.insert(stress, 0, [16.7])
    stress = np.insert(stress, stress.size, [17.7])
    plt.plot(stress, f(stress, m_value, b_value), "k--", label="Linear Fit")
    plt.legend(loc=3, borderaxespad=0.)

    plt.show()


def plot_strain_failure(initial_length, final_length, stress):
    failure_strain = []
    for ini, fin in zip(initial_length.values(), final_length.values()):
        failure_strain.append(abs(fin - ini) / ini)

    stress = [x / 10 ** 6 for x in stress]

    y_max = amax(failure_strain)
    x_max = amax(stress)

    spacing_x = x_max + x_max / 4
    spacing_y = y_max + y_max / 4

    plt.ylabel("Strain at Failure (%) {0} 2.54E-6 )".format(u"\u00B1"))
    plt.xlabel("Stress (MPa) {0} 9.81E-11)".format(u"\u00B1"))

    plt.axis([0, spacing_x, 0, spacing_y])

    plt.plot(stress, failure_strain, "ko", label="Failure Strain vs Failure Stress")

    (m_value, b_value, r_value, tt, stderr) = stats.linregress(stress, failure_strain)
    print(m_value, b_value)
    error = error_estimate(stress, failure_strain, m_value)

    plt.text(spacing_x * 0.05, spacing_y * 0.8,
             "Linear Fit Equation: \ny ={0:.4f}x {4} {3:.4f} + {1:.3f} %/MPa \nr = {2:.2f}".format(
                 m_value,
                 b_value,
                 r_value,
                 error,
                 u"\u00B1"))

    plt.plot(np.array([27] + stress), f(np.array([27] + stress), m_value, b_value), "k--", label="Linear Fit")
    plt.legend(loc=3, borderaxespad=0.)

    plt.show()


def create_graph(name):
    mass = {  # kg
        "Sample_1": 2.0195,
        "Sample_2": 2.1135,
        "Sample_3": 2.3674,
    }

    diameter = {  # mm
        "Sample_1": 0.94,
        "Sample_2": 0.956666,
        "Sample_3": 0.936666,
    }

    color_list = {
        "Sample_1": "ko",
        "Sample_2": "bo",
        "Sample_3": "ro",
    }

    stress = calculate_stress(mass, diameter)
    print(stress)

    start = [3, 6, 5]
    end = [8, 14, 11]

    initial_length = {
        "Sample_1": 45.98,
        "Sample_2": 67.04,
        "Sample_3": 44.33,
    }
    final_length = {
        "Sample_1": 56.82,
        "Sample_2": 87.38,
        "Sample_3": 61.75,
    }

    points = []
    reg_values = []
    for i in range(3):
        points.append(read_format_data(name[i], initial_length))
        reg_values.append(regression(points[i], start[i], end[i]))

    plot_strain_time(name, points, reg_values, start, end, color_list)
    plot_strain_rate_stress(reg_values, stress)
    plot_strain_failure(initial_length, final_length, stress)


create_graph(["Sample_1", "Sample_2", "Sample_3"])

plt.close("all")
