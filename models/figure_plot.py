import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, save_directory):
        self.save_directory = save_directory

    def plot_basic(self, data, title, xlabel, ylabel, file_name):
        plt.plot(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        self.save_plot(file_name)

    def plot_with_horizontal_line(self, data, line_value, line_label, title, xlabel, ylabel, file_name):
        plt.plot(data)
        plt.axhline(y = line_value, color = 'r', linewidth = 0.9, linestyle = '--', label = line_label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        self.save_plot(file_name)

    def plot_multiple_lines(self, data_list, labels, line_values, line_labels, title, xlabel, ylabel, file_name):
        for data, label in zip(data_list, labels):
            plt.plot(data, label = label)
        for line_value, line_label in zip(line_values, line_labels):
            plt.axhline(y = line_value, linewidth = 0.9, linestyle = '--', label = line_label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        self.save_plot(file_name)

    def save_plot(self, file_name):
        plt.savefig(f"{self.save_directory}/{file_name}", dpi = 300, bbox_inches = 'tight')
        plt.show()
        plt.close()
