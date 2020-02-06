import tkinter as tk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import locale
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fftpack import fft
from scipy.io.wavfile import read
from functools import partial
from tkinter import messagebox

matplotlib.use("TkAgg")
locale.setlocale(locale.LC_ALL, 'en_US')


def find_peak(data_in):
    """Returns maximum value of positive array of data"""

    currmax = 0.0
    for datapoint in data_in:
        if datapoint > currmax:
            currmax = datapoint
    return currmax


class InputData:
    def __init__(self):
        self.sample_rate = 0
        self.input_data = None
        self.raw_audio = None
        self.fundamental_freq = 0
        self.note_index = 0
        self.dynamics_index = 0
        self.sound_file_name = ""

    def load_sound_file(self, filename, config, main_app):
        """Opens specified .wav file for analysis"""

        self.sound_file_name = filename

        try:
            self.sample_rate, self.input_data = read("./sound_files/" + self.sound_file_name.get() + '.wav')
        except FileNotFoundError:
            messagebox.showerror("File not found", "The file \"" + self.sound_file_name.get()
                                 + ".wav\" could not be found")
        else:
            self.raw_audio = self.input_data.T[0]
            self.parse_file_name(main_app, config)
            main_app.ungrey_step()
            main_app.fft_plot.plot_fft(self, config, main_app)

    def parse_file_name(self, main_app, config):
        """Parses the file name to extract note, frequency, octave, and dynamics information"""

        name = self.sound_file_name.get()
        offset_note = config.lowest_note.get()

        # Splits file name into note, octave, and dynamics components
        i = 0
        while not name[i].isnumeric():
            i += 1
        note_string = name[0:i]
        octave = int(name[i])
        dynamics_string = name[(i + 1):]

        i = 0
        while not offset_note[i].isnumeric():
            i += 1
        offset_string = offset_note[0:i]
        offset_octave = int(offset_note[i])

        base_frequencies = {'C': 32.70, 'Cs': 34.65, 'D': 36.71, 'Ds': 38.89, 'E': 41.20, 'F': 43.65, 'Fs': 46.25,
                            'G': 49.00, 'Gs': 51.91, 'A': 55.00, 'As': 58.27, 'B': 61.74}
        base_indices = {'C': 0, 'Cs': 1, 'D': 2, 'Ds': 3, 'E': 4, 'F': 5, 'Fs': 6,
                        'G': 7, 'Gs': 8, 'A': 9, 'As': 10, 'B': 11}

        # Note index, frequency, and dynamics index are found
        # One is subtracted from the exponent because base_frequencies starts at C1, not C0
        try:
            self.fundamental_freq = base_frequencies[note_string] * (2 ** (octave - 1))
            self.note_index = base_indices[note_string] * (2 ** (octave - 1)) \
                              - base_indices[offset_string] * (2 ** (offset_octave - 1))
            self.dynamics_index = config.dynamics_dict[dynamics_string]
        except KeyError:
            messagebox.showerror("Invalid file name", "File does not follow naming conventions established in "
                                                      "configuration")


class Configuration:
    def __init__(self):
        self.dynamics_dict = {'pp': 0, 'p': 1, 'f': 2, 'ff': 3}
        self.lowest_note = None
        self.num_notes = 48
        self.num_partials = 30
        self.num_dynamics = 4
        self.num_times = 9
        self.graph_y_lim = 800

    def init_lowest_note(self):
        self.lowest_note = tk.StringVar()
        self.lowest_note.set('C1')


class Graph:
    def __init__(self, graph_frame):
        self.left_limit = 0
        self.right_limit = 5000  # TODO: TEMP*******
        self.tick = 5000  # REMOVE

        self.amplitude_data = None
        self.freq_range = None
        self.x_scale = 1
        self.time_index = 0

        f = plt.Figure(figsize=(5, 5), dpi=80)
        self.ax = f.subplots()
        self.ax.set_xlabel('Frequency')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_xlim(0, 20000)
        self.canvas = FigureCanvasTkAgg(f, graph_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def begin_tick(self, input_data, config, main_app):
        """Range of data is moved to beginning of sound file.  FFT is performed and plotted"""

        self.left_limit = 0
        self.tick = int(main_app.step_distance.get())

        if self.tick > input_data.raw_audio.size:
            self.right_limit = input_data.raw_audio.size
        else:
            self.right_limit = self.tick

        self.plot_fft(input_data, config, main_app)

    def back_tick(self, input_data, config, main_app):
        """Range of data is moved back by number of samples "tick" specified.  FFT is performed and plotted"""

        self.tick = int(main_app.step_distance.get())

        if (self.left_limit - self.tick) < 0:
            self.left_limit = 0
        else:
            self.left_limit = self.left_limit - self.tick

        if (self.left_limit + self.tick) > input_data.raw_audio.size:
            self.right_limit = self.tick
        else:
            self.right_limit = self.left_limit + self.tick

        self.plot_fft(input_data, config, main_app)

    def next_tick(self, input_data, config, main_app):
        """Range of data is moved forward by number of samples "tick" specified.  FFT is performed and plotted"""

        self.tick = int(main_app.step_distance.get())

        if (self.right_limit + self.tick) > input_data.raw_audio.size:
            self.right_limit = input_data.raw_audio.size
        else:
            self.right_limit = self.right_limit + self.tick

        if (self.right_limit - self.tick) < 0:
            self.left_limit = 0
        else:
            self.left_limit = self.right_limit - self.tick

        self.plot_fft(input_data, config, main_app)

    def end_tick(self, input_data, config, main_app):
        """Range of data is moved to end of sound file.  FFT is performed and plotted"""

        self.tick = int(main_app.step_distance.get())

        self.right_limit = input_data.raw_audio.size

        if self.tick > input_data.raw_audio.size:
            self.left_limit = 0
        else:
            self.left_limit = self.right_limit - self.tick

        self.plot_fft(input_data, config, main_app)

    def goto_tick(self, input_data, config, main_app):
        pass

    def plot_fft(self, input_data, config, main_app):
        """Transforms time domain sound data to frequency domain.
        Both x-axis (frequency) and y-axis (amplitude) are found and scaled appropriately
        """

        plot_data = input_data.raw_audio[self.left_limit:self.right_limit]
        plot_data_size = len(plot_data)
        full_length = len(input_data.raw_audio)

        # y-axis:  Performs fourier transform on audio data, and scales to account for data size
        self.amplitude_data = abs(fft(plot_data) / plot_data_size)

        # x-axis:  Finds frequencies to plot amplitude against so that data is scaled correctly
        self.x_scale = plot_data_size / input_data.sample_rate
        self.freq_range = np.arange(plot_data_size) / self.x_scale

        # y_range = find_peak(self.amplitude_data) * 1.2
        x_left_range = input_data.fundamental_freq * 0.5
        x_right_range = input_data.fundamental_freq * config.num_partials
        if x_right_range > 20000:
            x_right_range = 20000

        self.ax.clear()
        self.ax.plot(self.freq_range, self.amplitude_data, 'r')
        self.ax.set_xlim(x_left_range, x_right_range)
        self.ax.set_ylim(0, config.graph_y_lim)
        self.canvas.draw()

        main_app.status_message.set(f"{self.left_limit} : {self.right_limit} \t / {full_length}")


# str(self.left_limit) + ' : ' + str(self.right_limit) + '\t / ' + str(len(input_data.raw_audio))

class ArrayData:
    def __init__(self):
        # out_data[partial][note][dynamics][time]
        self.out_data = None

    def configure_array(self, config):
        pass

    def load_array(self, filename, config):
        """Loads in-progress output array data from a previous session so that it can be added to"""
        pass

    def analyze(self, input_data, graph, config):
        """Finds amplitude of each partial, places in appropriate array location"""

        for i in range(1, config.num_partials):
            left_index = int(input_data.fundamental_freq * (i - 0.3) * graph.x_scale)
            right_index = int(input_data.fundamental_freq * (i + 0.3) * graph.x_scale)

            self.out_data[i - 1][input_data.note_index][input_data.dynamics_index][graph.time_index] \
                = find_peak(graph.amplitude_data[left_index:right_index])

    def save_array(self, main_app, config):
        """Saves output array data to text file"""

        with open("./output_files/" + main_app.output_file_name + '.wav', 'w') as file_obj:

            file_obj.write(str(config.num_partials) + " " + str(config.num_notes) + " " +
                           str(config.num_dynamics) + " " + str(config.num_times))

            for harmonic in range(0, config.num_partials):
                for note in range(0, config.num_notes):
                    for dynamics in range(0, config.num_dynamics):
                        for time in range(0, config.num_times):
                            file_obj.write(str(int(self.out_data[harmonic][note][dynamics][time])) + " ")
                        file_obj.write("    ")
                    file_obj.write("\n")
                file_obj.write("\n")


class MainApplication:
    def __init__(self, input_data, config, array_data):
        # Main window setup
        self.main_window = tk.Tk()
        self.main_window.title('Pardessus Analyzer')
        self.main_window.geometry('1200x650')

        self.step_distance = tk.StringVar()
        self.sound_file_name = tk.StringVar()
        self.output_file_name = tk.StringVar()
        self.goto_position = tk.StringVar()
        config.init_lowest_note()
        self.index_num = "none"
        self.analyzed_list_string = "none"
        self.max_analyzed = "/9"  # "/" + config.num_times

        # ============================== Graph ============================== #
        self.graph_frame = tk.Frame(self.main_window)
        self.graph_frame.pack(fill=tk.X)
        self.fft_plot = Graph(self.graph_frame)

        # ========================== Control Panel ========================== #
        self.load_frame = tk.Frame(self.main_window)
        self.load_frame.pack(side=tk.LEFT)
        self.control_frame = tk.Frame(self.main_window, borderwidth=1)
        self.control_frame.pack()
        self.config_frame = tk.Frame(self.main_window)
        self.config_frame.pack()

        # ========================== Plotting Range ========================= #
        # Step navigation
        self.action_begin_tick = partial(self.fft_plot.begin_tick, input_data, config, self)
        self.action_prev_tick = partial(self.fft_plot.back_tick, input_data, config, self)
        self.action_next_tick = partial(self.fft_plot.next_tick, input_data, config, self)
        self.action_end_tick = partial(self.fft_plot.end_tick, input_data, config, self)

        self.step_label = tk.Label(self.control_frame, text="Step:")
        self.step_label.grid(row=0, column=3, sticky=tk.W, pady=6)
        self.step_entry = tk.Entry(self.control_frame, width=8, textvariable=self.step_distance)
        self.step_entry.insert(tk.END, '5000')
        self.step_entry.grid(row=0, column=4, sticky=tk.W)
        self.samples_label = tk.Label(self.control_frame, text="samples")
        self.samples_label.grid(row=0, column=5, sticky=tk.W)
        self.step_blank = tk.Label(self.control_frame)
        self.step_blank.grid(row=0, column=6, padx=15)
        self.btn_begin = tk.Button(self.control_frame, state=tk.DISABLED, text='<<', width=5,
                                   command=self.action_begin_tick)
        self.btn_begin.grid(row=0, column=7)
        self.btn_prev = tk.Button(self.control_frame, state=tk.DISABLED, text='<', width=5,
                                  command=self.action_prev_tick)
        self.btn_prev.grid(row=0, column=8)
        self.btn_next = tk.Button(self.control_frame, state=tk.DISABLED, text='>', width=5,
                                  command=self.action_next_tick)
        self.btn_next.grid(row=0, column=9)
        self.btn_end = tk.Button(self.control_frame, state=tk.DISABLED, text='>>', width=5,
                                 command=self.action_end_tick)
        self.btn_end.grid(row=0, column=10)

        # ============================ Analysis ============================= #
        self.analyze_label = tk.Label(self.control_frame, text="Analyze:")
        self.analyze_label.grid(row=1, column=3, sticky=tk.W, pady=6)
        self.index_label = tk.Label(self.control_frame, text="index: ")
        self.index_label.grid(row=1, column=4, sticky=tk.W)
        self.index_num_label = tk.Label(self.control_frame, text=self.index_num)
        self.index_num_label.grid(row=1, column=5, sticky=tk.W)
        self.btn_capture = tk.Button(self.control_frame, state=tk.DISABLED, text="capture", width=5)
        self.btn_capture.grid(row=1, column=7)
        self.btn_prev_index = tk.Button(self.control_frame, state=tk.DISABLED, text="<", width=5)
        self.btn_prev_index.grid(row=1, column=8)
        self.btn_next_index = tk.Button(self.control_frame, state=tk.DISABLED, text=">", width=5)
        self.btn_next_index.grid(row=1, column=9)
        self.btn_delete = tk.Button(self.control_frame, state=tk.DISABLED, text="delete", width=5)
        self.btn_delete.grid(row=1, column=10)
        self.captured_label = tk.Label(self.control_frame, text="analyzed:")
        self.captured_label.grid(row=2, column=4)
        self.analyzed_list_display = tk.Label(self.control_frame, text=self.analyzed_list_string)
        self.analyzed_list_display.grid(row=2, column=5, columnspan=4, sticky=tk.W)
        self.max_analyzed_label = tk.Label(self.control_frame, text=self.max_analyzed)
        self.max_analyzed_label.grid(row=2, column=9, sticky=tk.W)

        # ============================== Output ============================= #
        self.action_new_data = partial(self.ungrey_analysis)

        self.output_file_label = tk.Label(self.control_frame, text="Output file:")
        self.output_file_label.grid(row=4, column=3, sticky=tk.W, pady=6)
        self.output_file_name_entry = tk.Entry(self.control_frame, width=8, textvariable=self.output_file_name)
        self.output_file_name_entry.grid(row=4, column=4, sticky=tk.W)
        self.txt_label = tk.Label(self.control_frame, text=".txt")
        self.txt_label.grid(row=4, column=5, sticky=tk.W)
        self.btn_load_out_file = tk.Button(self.control_frame, text="load", width=5)
        self.btn_load_out_file.grid(row=4, column=7)
        self.btn_save_out_file = tk.Button(self.control_frame, text="save", width=5)
        self.btn_save_out_file.grid(row=4, column=8)
        self.btn_new_data = tk.Button(self.control_frame, text="new", width=5, command=self.action_new_data)
        self.btn_new_data.grid(row=4, column=10)

        # ============================ Sound File =========================== #
        self.action_load_sound_file = partial(input_data.load_sound_file, self.sound_file_name,
                                              config, self)

        self.sound_file_label = tk.Label(self.control_frame, text="Sound file:")
        self.sound_file_label.grid(row=5, column=3, sticky=tk.W, pady=6)
        self.sound_file_entry = tk.Entry(self.control_frame, width=8, textvariable=self.sound_file_name)
        self.sound_file_entry.grid(row=5, column=4, sticky=tk.W)
        self.wav_label = tk.Label(self.control_frame, text=".wav")
        self.wav_label.grid(row=5, column=5, sticky=tk.W)
        self.btn_load_sound_file = tk.Button(self.control_frame, text="load", width=5,
                                             command=self.action_load_sound_file)
        self.btn_load_sound_file.grid(row=5, column=7, sticky=tk.W)

        # ============================ Status Bar =========================== #
        self.status_message = tk.StringVar()
        self.status = tk.Label(self.main_window, textvariable=self.status_message, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        self.main_window.mainloop()

    def ungrey_step(self):
        """Enables controls for going through fft graph"""
        self.btn_begin.configure(state=tk.NORMAL)
        self.btn_prev.configure(state=tk.NORMAL)
        self.btn_next.configure(state=tk.NORMAL)
        self.btn_end.configure(state=tk.NORMAL)

    def ungrey_analysis(self):
        """Enables controls for analysis of fft"""
        self.btn_capture.configure(state=tk.NORMAL)
        self.btn_prev_index.configure(state=tk.NORMAL)
        self.btn_next_index.configure(state=tk.NORMAL)
        self.btn_delete.configure(state=tk.NORMAL)


_config = Configuration()
_input_data = InputData()
_array_data = ArrayData()

pardessus_main = MainApplication(_input_data, _config, _array_data)

# TODO:  String variables like 'filename' may need to be converted from a tkinter string object
# Ability to play audio starting at current point
# Configuration window, data smoothing window


# class Configuration:
#     pass




