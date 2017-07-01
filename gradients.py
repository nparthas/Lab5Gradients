from scipy import stats
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


def regression(points, start, end):
    (m_value, b_value, r_value, tt, stderr) = stats.linregress(points["Time"][start:end],
                                                               points["Strain"][start:end])
    print('regression: a=%.4f b=%.4f, r=%.2f, std error= %.3f' % (m_value, b_value, r_value, stderr))
    return [m_value, b_value, r_value, stderr]


def f(x, m, b):
    return m * x + b


def plot_values(title_name, points, reg_values):
    color_list = {
        "Sample_1": "ko",
        "Sample_2": "bo",
        "Sample_3": "ro",
    }

    x_max = amax(points["Time"])
    y_max = amax(points["Strain"])

    spacing_x = x_max + x_max / 5
    spacing_y = y_max + y_max / 5

    plt.plot(points["Time"], points["Strain"], color_list[title_name])

    plt.title("Force-Deflection Plot for " + title_name.replace("_", " "))

    plt.xlabel("Time (s {} 0.001)".format(u"\u00B1"))
    plt.ylabel("Strain (% {} 0.02)".format(u"\u00B1"))

    plt.axis([0, spacing_x, 0, spacing_y])
    plt.text(spacing_x * 0.05, spacing_y * 0.8,
             "Linear Fit Equation: \ny ={0:.3f}x + {1:.3f} N/m \nr = {2:.2f} \nstd error = {3:.2f}".format(
                 reg_values[0],
                 reg_values[1],
                 reg_values[2],
                 reg_values[3]))

    plt.plot(points["Time"], f(points["Time"], reg_values[0], reg_values[1]),
             color_list[title_name].replace("o", "--"), label="Linear Fit")

    plt.legend(loc=4, borderaxespad=0.)

    plt.show()


def create_graph(name, start, end):
    points = read_format_data(name)
    reg_values = regression(points, start, end)
    plot_values(name, points, reg_values)


create_graph("Sample_1", 0, 9)

create_graph("Sample_2", 0, 16)

create_graph("Sample_3", 0, 15)

plt.close("all")
