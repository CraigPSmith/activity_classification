
def kte(signal):

    for ii in range(0, len(signal) - (40 - 1)):


        d2 = signal[1:-1].reset_index(drop=True)**2

        d3 = signal[:-2].reset_index(drop=True) * signal[2:].reset_index(drop=True)
        kte = d2-d3

    return kte