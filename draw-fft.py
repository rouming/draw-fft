#!/usr/bin/python3
"""
Draw FFT
"""
import sys
from types import SimpleNamespace
import arcade as ar
from arcade.experimental.uislider import UISlider
from arcade.gui import UIManager, UIAnchorWidget, UILabel, UIFlatButton
from arcade.gui.events import UIOnChangeEvent, UIOnClickEvent
import numpy as np
import datetime
import soundfile as sf
import random
from optparse import OptionParser
from PIL import Image
from svgpathtools import svg2paths, path
from tsp_solver.greedy_numpy import solve_tsp

SCREEN_TITLE = "Draw FFT"
# Don't show info about frequencies with magnitute below this threshold
MAG_THRESH = 0.1
# Don't show circles with radious below this threshold. This dramatically
# optimizes drawing on number of frequencies > 1000
RADIUS_THRESH = 2.0
BLACK_THRESHOLD = 128

AMPL_SCALE = SimpleNamespace(vmin=50,  vmax=500,  value=200)
FREQ_SCALE = SimpleNamespace(vmin=0,   vmax=4,    value=0.5)
TIME_SCALE = SimpleNamespace(vmin=0,   vmax=3,    value=1)
MAX_POINTS = SimpleNamespace(vmin=0,   vmax=2000, value=1000)

class freq():
    def __init__(self, f, phase, ampl):
        self.f = f
        self.phase = phase
        self.ampl = ampl

    def draw(self, dt, center):
        # Calculate rotation vector. Negative sign to keep images (2-dim)
        # represented correctly and not upside down. This happens because
        # both vector and raster functions produce points in a Cartesian
        # pixel coordinate system, with (0,0) in the top left corner.
        # Arcade is different and has (0,0) in the bottom left corner.

        p = np.exp(-1j * (self.phase + dt * self.f))
        p = np.array([p.real, p.imag])

        # Scale radius
        radius = AMPL_SCALE.value * self.ampl
        p = center + p * radius

        if radius > RADIUS_THRESH:
            # Don't spend resources on almost invisible circles.
            # This dramatically increases performance on number
            # of frequencies > 1000

            c1 = (200, 200, 200, 100)
            c2 = (100, 100, 100, 100)
            c3 = (150, 0, 0, 100)

            ar.draw_circle_outline(center[0], center[1],  radius, c1)
            ar.draw_line(center[0], center[1], p[0], p[1], c2, 2)
            ar.draw_circle_filled(center[0], center[1], 4.0, c3)

        return p

class draw_fft(ar.Window):
    def setup_ui(self, is_1_dim):
        sz = ar.get_display_size(0)
        super().__init__(sz[0], sz[1], SCREEN_TITLE)
        ar.set_background_color(ar.csscolor.CORNFLOWER_BLUE)

        # Required, create a UI manager to handle all UI widgets
        self.manager = UIManager()
        self.manager.enable()

        # Frequncy slider placement
        freq_slider = UISlider(min_value=FREQ_SCALE.vmin,
                               max_value=FREQ_SCALE.vmax,
                               value=FREQ_SCALE.value,
                               width=500, height=50)
        @freq_slider.event()
        def on_change(event: UIOnChangeEvent):
            FREQ_SCALE.value = freq_slider.value

        self.manager.add(UIAnchorWidget(anchor_x="right", anchor_y="top",
                                        align_x=-20,
                                        child=freq_slider))
        self.manager.add(UIAnchorWidget(anchor_x="right", anchor_y="top",
                                        align_x=-freq_slider.width - 40,
                                        align_y=-15,
                                        child=UILabel(text=f"Frequency")))

        # Amplitude slider placement
        ampl_slider = UISlider(min_value=AMPL_SCALE.vmin,
                               max_value=AMPL_SCALE.vmax,
                               value=AMPL_SCALE.value,
                               width=500, height=50)
        @ampl_slider.event()
        def on_change(event: UIOnChangeEvent):
            AMPL_SCALE.value = ampl_slider.value

        self.manager.add(UIAnchorWidget(anchor_x="right", anchor_y="top",
                                        align_x=-20,
                                        align_y=-50,
                                        child=ampl_slider))
        self.manager.add(UIAnchorWidget(anchor_x="right", anchor_y="top",
                                        align_x=-ampl_slider.width - 40,
                                        align_y=-ampl_slider.height - 15,
                                        child=UILabel(text=f"Amplitude",
                                        align_y=100)))

        # Points history slider placement
        hist_slider = UISlider(min_value=MAX_POINTS.vmin,
                               max_value=MAX_POINTS.vmax,
                               value=MAX_POINTS.value,
                               width=500, height=50)
        @hist_slider.event()
        def on_change(event: UIOnChangeEvent):
            MAX_POINTS.value = int(hist_slider.value)

        self.manager.add(UIAnchorWidget(anchor_x="right", anchor_y="top",
                                        align_x=-20,
                                        align_y=-100,
                                        child=hist_slider))
        self.manager.add(UIAnchorWidget(anchor_x="right", anchor_y="top",
                                        align_x=-ampl_slider.width - 40,
                                        align_y=-ampl_slider.height - 65,
                                        child=UILabel(text=f"History",
                                        align_y=100)))


        if is_1_dim:
            # 1-dim time slider placement
            time_slider = UISlider(min_value=TIME_SCALE.vmin,
                                   max_value=TIME_SCALE.vmax,
                                   value=TIME_SCALE.value,
                                   width=500, height=50)
            @time_slider.event()
            def on_change(event: UIOnChangeEvent):
                TIME_SCALE.value = time_slider.value

            self.manager.add(UIAnchorWidget(anchor_x="right", anchor_y="top",
                                            align_x=-20,
                                            align_y=-150,
                                            child=time_slider))
            self.manager.add(UIAnchorWidget(anchor_x="right", anchor_y="top",
                                            align_x=-ampl_slider.width - 40,
                                            align_y=-ampl_slider.height - 115,
                                            child=UILabel(text=f"X axis (time)",
                                                          align_y=100)))

        reset = UIFlatButton(text="Reset", width=100)
        @reset.event()
        def on_click(event: UIOnClickEvent):
            self.points = np.empty((0,2))

        self.manager.add(UIAnchorWidget(anchor_x="right", anchor_y="top",
                                        align_x=-160,
                                        align_y=-220,
                                        child=reset))

        quitt = UIFlatButton(text="Quit", width=100)
        @quitt.event()
        def on_click(event: UIOnClickEvent):
            ar.exit()

        self.manager.add(UIAnchorWidget(anchor_x="right", anchor_y="top",
                                        align_x=-35,
                                        align_y=-220,
                                        child=quitt))


    def __init__(self, is_1_dim, fft_data, fft_freq):
        # Setup UI
        self.setup_ui(is_1_dim)

        # Get magnitude and normalize
        fft_mag = np.abs(fft_data)
        fft_mag = fft_mag / np.linalg.norm(fft_mag)
        # Get phase
        fft_phase = np.angle(fft_data)

        mag_indices = [{"mag": mag, "ind": i} for i, mag in
                       enumerate(fft_mag)]
        sorted_indices = [each['ind'] for each in
                          sorted(mag_indices, key=lambda each: each['mag'],
                                 reverse=True)]

        # Select magnitude higher than threshold
        ind = np.argwhere(fft_mag > MAG_THRESH)
        self.mag_thresh = fft_mag[ind].flatten()
        self.freq_thresh = fft_freq[ind].flatten()

        self.dt = 0
        self.is_1_dim = is_1_dim
        self.points = np.empty((0,2))
        self.freqs = []
        for i in sorted_indices:
            (f, phase, mag) = (fft_freq[i], fft_phase[i], fft_mag[i])
            if is_1_dim:
                # Swap sin/cos in order to get correct representation on the
                # horizontal axis (time) of the 1-dim signal. Once inverse
                # fourier transform is calculated the real part (x axis)
                # represents the recovered signal, but the drawing routine
                # takes imaginary part (y axis) as a signal (time is the
                # horizontal axis, not vertical), so in order to be
                # consistent provide a phase shift here. Why positive and
                # not negative? Well, see the corresponding comment about
                # the negative sign in exponent of the rotation vector in
                # the draw() function.
                phase += np.pi/2
            self.freqs.append(freq(f, phase, mag))

    def setup(self):
        pass

    def on_draw(self):
        """ Render the screen. """
        ar.start_render()
        w, h = self.get_size()

        # Draw found frequencies
        text_y = h/2
        for f in self.freq_thresh:
            text = "%3.3fHz" % f
            ar.draw_text(text, w/2, text_y, ar.color.WHITE,
                         10, width=300)
            text_y -= 20

        start_p = p = np.array([w/2 - 400, h/2])
        self.dt += 0.01 * FREQ_SCALE.value

        # Draw each frequency circle
        for freq in self.freqs:
            p = freq.draw(self.dt, p)

        if self.is_1_dim:
            # Draw projection line
            end = np.array([start_p[0] + 300, p[1]])
            ar.draw_line(p[0], p[1], end[0], end[1], ar.color.BLACK, 1)
            p = end

        # Limit points
        if len(self.points) > MAX_POINTS.value:
            self.points = self.points[len(self.points) - MAX_POINTS.value:]

        # Pick up a point
        self.points = np.append(self.points, [p], axis=0)

        if self.is_1_dim:
            # Simulate time for 1-dim case by shifting all points to the right.
            # For the 2-dim case time is Z-axis, which points to the screen,
            # which we don't see
            self.points += (1.0 * TIME_SCALE.value, 0)

        # Draw the whole points list
        for p in self.points:
            r = 4.0
            ar.draw_circle_filled(p[0], p[1], r, ar.color.GREEN,
                                  num_segments=int(2.0 * np.pi * r / 3.0))

        # Draw UI
        self.manager.draw()


# Slow fourier transform :)
def sft(sig):
    n = len(sig)
    zeta = np.exp(-2 * np.pi * 1j / n)
    freq = np.array([np.array([sig[i] * zeta**(i * f) for i in range(0, n)]).sum()
            for f in range(0, n)])
    return freq

def image_to_points(image, num, threshold):
    h, w = image.shape
    black_pixels_n = (image < threshold).sum()
    assert black_pixels_n > num, "No pixels of specified threshold in the image!"
    points = np.zeros(num, dtype=complex)
    for i in range(num):
        while True:
            x = int(random.random() * w)
            y = int(random.random() * h)
            # Black threshold
            if image[y, x] < threshold:
                break
        # Convert to complex
        points[i] = x + y*1j
    return points

def distance_matrix(points):
    m = np.array([points] * len(points))
    return np.abs(m - m.T)

def read_raster_image(filename):
    image = Image.open(filename).convert("RGBA")
    # Alpha as white
    bg = Image.new("RGBA", image.size, (255,255,255))
    # Greyscale, stores only 'L'uminance
    image = Image.alpha_composite(bg, image).convert("L")
    return np.asarray(image)

def process_raster_file(filename, num):
    image = read_raster_image(filename)
    points = image_to_points(image, num, BLACK_THRESHOLD)
    dist = distance_matrix(points)
    indices = solve_tsp(dist)
    return points[indices]

def process_vector_file(filename, num):
    paths, _ = svg2paths(filename)
    paths = path.concatpaths(paths)
    time = np.linspace(0.0, 1.0, num)
    points = np.array([paths.point(t) for t in time], dtype=complex)
    return points

def process_audio_file(filename, num):
    data, rate = sf.read(filename)[:num]
    if len(data.shape) > 1:
        # Only first channel
        data = data[:, 0]
    return data, rate

def process_file_and_fft(filename, options):
    ext = filename.split(".")[-1]
    is_1_dim = False
    if ext == "wav":
        points, rate = process_audio_file(filename, options.samples_count)
        # Audio signal is 1-dim, so we are not interested in negative
        # frequencies
        start = len(points) // 2
        is_1_dim = True
    elif ext in ("png", "jpeg", "jpg", "svg"):
        if ext == "svg":
            points = process_vector_file(filename, options.samples_count)
        else:
            points = process_raster_file(filename, options.samples_count)
        # Be aware that an array from both vector and raster functions are
        # are complex numbers represent points in a Cartesian pixel coordinate
        # system, with (0,0) in the top left corner. Arcade is different and
        # has (0,0) in the bottom left corner. To deal with that we don't
        # invert an imaginary part (y axis), but instead invert rotation
        # of the fourier unit vector.

        # Assume 1s duration
        rate = len(points)
        # 2-dim signal, take negative frequencies into account
        start = len(points) // 2 - options.freq_count // 2
    else:
        assert False, "Unknown file extension!"

    end = start + options.freq_count

    # FFT calculation
    fft_freq = np.fft.fftfreq(len(points), d=1./rate)
    fft_data = np.fft.fft(points)
    # Drop 0 frequency
    fft_freq = fft_freq[1:]
    fft_data = fft_data[1:]
    # Orgainize frequencies naturally: from negative to positive
    fft_freq = np.fft.fftshift(fft_freq)
    fft_data = np.fft.fftshift(fft_data)
    # Slice
    fft_freq = fft_freq[start:end]
    fft_data = fft_data[start:end]

    return is_1_dim, fft_data, fft_freq


def main():
    usage = "Usage: %prog [OPTIONS] FILE"
    parser = OptionParser(usage=usage,
                          description="Draws an image or a signal from an audio (.wav), raster (.png, .jpeg) or vector (.svg) FILE using DTTF")
    parser.add_option("-p", "--samples-count", type="int", default=1000,
                      dest="samples_count", help="Number of samples extracted from input image or audio")
    parser.add_option("-c", "--freq-count", type="int", default=1000,
                      dest="freq_count", help="Number of frequencies")
    options, args = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)
    else:
        filename = args[0]

    is_1_dim, fft_data, fft_freq = process_file_and_fft(filename, options)

    window = draw_fft(is_1_dim, fft_data, fft_freq)
    window.setup()
    ar.run()

if __name__ == "__main__":
    main()
