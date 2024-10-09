import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
from collections import namedtuple
from enum import Enum

Header = namedtuple('Header', ['event_size', 'board_id', 'board_fail_flag',
                               'pattern', 'group_mask', 'event_counter', 'trigger_time_tag', 'ro_flag'])


class PlotOption(Enum):
    SINGLE_EVENT = 0
    MULTIPLE_EVENT = 1


def get_time_values(TTTs, E, wvfm_size, ns_per_sample=16):
    to_ns = 8
    c_unit = 1E-3  # from ns to us
    clock_reset = 1E6 * E  # us

    ttt_ini = (TTTs[E] * to_ns - wvfm_size * ns_per_sample) * c_unit + clock_reset  # us
    ttt_end = (TTTs[E] * to_ns * c_unit + clock_reset)  # us

    time = np.linspace(ttt_ini, ttt_end, wvfm_size)

    return time


def plot_from_wvfm(time, wvfm, E=-1, CH=-1):
    plt.scatter(time, wvfm, color='blue', s=1)
    plt.ylabel("ADCs")
    plt.xlabel("Time [us]")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useOffset=False, useMathText=True)
    if E != -1 and CH != -1:
        plt.title(f"CH {CH + 1} EVENT {E + 1}")
    plt.show()


def plot_from_events(E, CH, events, TTTs):
    wvfm = events[E][CH]
    time = get_time_values(TTTs=TTTs, E=E, wvfm_size=wvfm.size)

    plot_from_wvfm(time=time, wvfm=wvfm, E=E, CH=CH)


def plot_events_single_channel(TTTs, events, CH, E_ini=1, E_end=100):
    wvfm = np.array([])
    time = np.array([])

    E = E_ini

    for wvfms in events[E_ini:E_end]:
        wvfm = np.append(wvfm, wvfms[CH])
        time = np.append(time, get_time_values(TTTs=TTTs, E=E, wvfm_size=wvfms[CH].size))

        plot_from_wvfm(time=time, wvfm=wvfm, E=E, CH=CH)

        E += 1


def aux_check_number_events(f, logger, num_events, num_words_per_event):
    counter = 0
    last_i = 0
    num_words = num_events * num_words_per_event
    for i in range(num_words):
        word = read_word(f)
        if get_range(word, 27, 0) == num_words_per_event: # 15268
            diff = i - last_i
            logger.info(f"header starting at: {i} \t {diff}")
            counter += 1
            last_i = i
    logger.info(f"Number of events: {counter}")

def aux_check_ttt_diff(f, logger):
    NUM_WORDS_PER_HEADER = 4

    last_ttt = 0
    last_wc = 0
    wc = 0

    # Loop variables and events data
    can_continue = True
    e = 0

    # Loop over all events which collects the waveform data
    while e < max_e and can_continue:
        try:
            header = read_header(f=f, verbose=verbose, logger=logger)

            NUM_WORDS_PER_EVENT = header.event_size
            NUM_WORDS_PER_WVFM = NUM_WORDS_PER_EVENT - NUM_WORDS_PER_HEADER

            # TTT logging info
            ttt = header.trigger_time_tag
            ttt_diff = ttt - last_ttt
            c_diff = wc - last_wc
            logger.info(
                f"Event {header.event_counter} header starting at: {wc}\t{c_diff}\t"
                f"{ttt * 8} ns = {(ttt * 8) * 1E-9} s\t{ttt_diff * 8} ns"
            )
            last_ttt = ttt
            last_wc = wc
            wc += 4

            for j in range(NUM_WORDS_PER_WVFM):
                wc += 1
                read_word(f)

        except ValueError as exc:
            logger.warning(f"Trying to read event {e + 1}: {exc.args[0]}")
            can_continue = False

        e += 1
def read_word(f):
    """ Read 32-bits word. """
    word = f.read(4)
    if len(word) < 4:
        raise ValueError("Not enough data to read a 32-bit word.")
    word = int.from_bytes(word, byteorder="little")
    return word


def read_sample(f):
    """ Read 16-bits sample. """
    sample = f.read(2)
    if len(sample) < 2:
        raise ValueError("Not enough data to read a 32-bit word.")
    sample = int.from_bytes(sample, byteorder="little")
    return sample


def get_range(word, msb, lsb):
    """ Read the encoded information in the given a word and an interval. """
    mask = (1 << (msb - lsb + 1)) - 1
    shifted_word = word >> lsb
    result = shifted_word & mask
    return result


def read_header(f, log_file, logger):
    # Read the 4 first words (the header)
    header_words = []
    for i in range(4):
        word = read_word(f)
        header_words.append(word)
        logging.debug(header_words[i])

    # 1st word
    # - Event size: number of 32-bit long words to be read.
    event_size = get_range(header_words[0], 27, 0)

    # 2nd word
    # - Board ID: GEO address (meaningful for VME64X modules).
    board_id = get_range(header_words[1], 31, 27)
    # - Board Fail Flag: 1 if there is a hardware problem. For more info read register address 0x8178.
    board_fail_flag = get_range(header_words[1], 26, 26)
    # - Pattern: 16-bit pattern latched on the LVDS I/Os as the trigger arrives.
    pattern = get_range(header_words[1], 23, 8)
    # - Group mask: mask of the groups participating in the event.
    group_mask = get_range(header_words[1], 7, 0)

    # 3rd word
    # - Event counter: it can count either accepted trigger or all triggers (bit[3] of register address 0x8100).
    event_counter = get_range(header_words[2], 23, 0)

    # 4th word
    # - Trigger time tag: trigger time reference.
    trigger_time_tag = get_range(header_words[3], 30, 0)
    ro_flag = get_range(header_words[3], 31, 31)

    header = Header(event_size, board_id, board_fail_flag, pattern, group_mask, event_counter, trigger_time_tag,
                    ro_flag)

    if log_file:
        logger.info("\n================================================\n"
                     "#################### HEADER ####################\n"
                     "================================================\n"
                     "=================== 1st WORD ===================\n"
                     f"Event size: {header.event_size}\n"
                     "=================== 2nd WORD ===================\n"
                     f"Board ID: {header.board_id}\n"
                     f"Board Fail Flag: {header.board_fail_flag}\n"
                     f"Pattern: {header.pattern}\n"
                     f"Group Mask: {header.group_mask}\n"
                     "=================== 3rd WORD ===================\n"
                     f"Event Counter: {header.event_counter}\n"
                     "=================== 4th WORD ===================\n"
                     f"Trigger time tag: {header.trigger_time_tag * 8} ns = {(header.trigger_time_tag * 8) * pow(10, -9)} s\n"
                     f"Roll-over flag: {header.ro_flag}\n"
                     "================================================\n"
                     )
    return header

def read_events(f, logger, log_file, max_e):
    # Constants
    WORD_BITS = 32
    SAMPLE_BITS = 12
    NUM_WORDS_PER_HEADER = 4
    NUM_WVFMS = 64

    # Read and decode the header of an event
    last_ttt = 0
    last_wc = 0
    wc = 0

    # Loop variables and events data
    can_continue = True
    e = 0
    events = []
    TTTs = []

    # Loop over all events which collects the waveform data
    while e < max_e and can_continue:
        try:
            header = read_header(f=f, log_file=log_file, logger=logger)

            NUM_WORDS_PER_EVENT = header.event_size
            NUM_WORDS_PER_WVFM = NUM_WORDS_PER_EVENT - NUM_WORDS_PER_HEADER
            NUM_SAMPLES_PER_WVFM = NUM_WORDS_PER_WVFM * WORD_BITS // SAMPLE_BITS // NUM_WVFMS
            NUM_SAMPLES_PER_GROUP = 8 * NUM_SAMPLES_PER_WVFM

            logger.info(f"Header event size: {NUM_WORDS_PER_EVENT}")
            logger.info(f"Words per header: {NUM_WORDS_PER_HEADER}")
            logger.info(f"Word per wvfm: {NUM_WORDS_PER_WVFM}")
            logger.info(f"Samples per wvfm: {NUM_SAMPLES_PER_WVFM}")
            logger.info(f"Samples per group: {NUM_SAMPLES_PER_GROUP}")

            # TTT logging info
            ttt = header.trigger_time_tag
            TTTs.append(ttt)
            ttt_diff = ttt - last_ttt
            c_diff = wc - last_wc
            logger.info(
                f"Event {header.event_counter} header starting at: {wc}\t{c_diff}\t"
                f"{ttt * 8} ns = {(ttt * 8) * 1E-9} s\t{ttt_diff * 8} ns"
            )
            last_ttt = ttt
            last_wc = wc
            wc += 4

            # Set the wvfm matrix.
            wvfms = np.zeros((NUM_WVFMS, NUM_SAMPLES_PER_WVFM))
            S = 0  # Absolute sample index.

            buffer = 0
            bits_in_buffer = 0
            for j in range(NUM_WORDS_PER_WVFM):
                wc += 1
                word = read_word(f)
                # Add new 32 bits to the buffer
                buffer |= word << bits_in_buffer
                bits_in_buffer += WORD_BITS

                # Extract 12-bit sequences from the buffer
                while bits_in_buffer >= SAMPLE_BITS:
                    c = ((S // 3) % 8) + (S // NUM_SAMPLES_PER_GROUP) * 8  # Channel index
                    s = (S % 3) + ((S//24) * 3) % NUM_SAMPLES_PER_WVFM  # Sample/channel index
                    sample = get_range(buffer, SAMPLE_BITS - 1, 0)

                    if log_file:
                        logger.info(f"Sample: {sample} \t g {(S // NUM_SAMPLES_PER_GROUP)} ch {c} s {s} S {S}")

                    wvfms[c][s] = sample

                    buffer >>= SAMPLE_BITS
                    bits_in_buffer -= SAMPLE_BITS
                    S += 1

            events.append(wvfms)

            if bits_in_buffer > 0:
                logger.warning(f"{bits_in_buffer} bits left in buffer which cannot form a full 12-bit word.")

        except ValueError as exc:
            logger.warning(f"Trying to read event {e + 1}: {exc.args[0]}")
            can_continue = False

        e += 1

    return events, TTTs


def plot_events(logger, events, TTTs, plot_option, E_ini, E_end):
    if plot_option == PlotOption.SINGLE_EVENT:
        plot_from_events(E=0, CH=0, events=events, TTTs=TTTs)
    elif plot_option == PlotOption.MULTIPLE_EVENT:
        plot_events_single_channel(TTTs=TTTs, events=events, CH=0, E_ini=E_ini, E_end=E_end)


if __name__ == "__main__":

    # Access the binary file
    DATA_PATH = "data/XARAPUCAdecoder/input/rawbin_V1740_2024.10.01-12.18.17.dat"
    f = open(DATA_PATH, mode="rb")

    # Configure arguments
    parser = argparse.ArgumentParser(description="Binary decoder for CAEN events from the V1740B digitizer.")

    parser.add_argument("-N", type=int, required=True, help="Number of events to be processed starting from 0.")

    parser.add_argument("-P", type=int, default=PlotOption.SINGLE_EVENT.value, choices=[option.value for option in PlotOption],
                        required=True, help="Plotting option: 0 for single event plot, 1 for multiple event plots.")
    parser.add_argument("-E_ini", type=int, default=1, help="Initial event index for multiple event plot option.",
                        required=False)
    parser.add_argument("-E_end", type=int, default=99, help="Final event index for multiple event plot option.",
                        required=False)

    parser.add_argument("-L", action="store_true", help="Activates logging node in \"decoder.log\" file.",
                        required=False)

    parser.add_argument("-V", action="store_true", help="Activates verbose mode in console.",
                        required=False)

    parser.add_argument("-CE", action="store_true", help="Call function to check number of events.",
                        required=False)
    parser.add_argument("-W",type=int, default=0, help="Set the number of words to check for -CE option.",)

    parser.add_argument("-CT", action="store_true", help="Call function to check TTT of the events.",
                        required=False)

    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    max_e = args.N
    plot_option = PlotOption(args.P)
    E_ini = args.E_ini
    E_end = args.E_end
    log_file = args.L
    verbose = args.V
    check_number_events = args.CE
    check_ttt = args.CT
    check_num_words = args.W

    # Configure a logger
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if log_file:
        # Configure and activate logging in a file
        file_handler = logging.FileHandler("decoder.log", mode="w")
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    if verbose:
        # Configure and activate verbose in console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)

    # Logging information
    if log_file:
        logger.info("Logging activated, saving to file.")
    elif verbose:
        logger.info("Logging activated in verbose mode, showing in console.")

    logger.info(f"Scanning the first {max_e + 1} events from file \"{DATA_PATH}\".")

    logger.info(f"Plotting option: {plot_option}")
    if plot_option == PlotOption.MULTIPLE_EVENT:
        logger.info(f"Multiple events plotting limits set to E_init: {E_ini}, E_end: {E_end}")

    # Execute main function given a configuration
    if check_number_events:
        aux_check_number_events(f=f, logger=logger, num_events=max_e, num_words_per_event=check_num_words)
    elif check_ttt:
        aux_check_ttt_diff(f=f, logger=logger)
    else:
        # Decode binary file.
        events, TTTs = read_events(f=f, logger=logger, log_file=log_file, max_e=max_e)
        plot_events(logger=logger, events=events, TTTs=TTTs, plot_option=plot_option, E_ini=E_ini, E_end=E_end)

    f.close()
