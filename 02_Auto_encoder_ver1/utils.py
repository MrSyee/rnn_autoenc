import numpy as np
from midi import utils as midiutils

class Util:
    def __init__(self):
        '''
        set index <-> data
        pitch: C3 ~ C#6 (48 ~ 85)
        duration: [4.0, 3.5(..), 3.0, 2.0, 1.75(..), 1.5, 1.0(quarter), 0.75, 0.5, 0.25]
        '''
        # MIDI file path
        self.MIDI_PATH = "./midi/songs/"
        self.FILE_LIST = midiutils.load_filename(self.MIDI_PATH)  # all .mid file list

        # load MIDI file
        self.all_song = midiutils.load_all_midi(self.FILE_LIST, self.MIDI_PATH)

        self.pitch_sample = list(range(48, 86))
        self.pitch_sample.append(0) # start
        self.pitch_sample.append('Rest')

        self.duration_sample = [4.0, 3.5, 3.0, 2.0, 1.75, 1.5, 1.0, 0.75, 0.5, 0.25]
        self.duration_sample.append(0)  # start

    def get_one_song(self, filename):
        return midiutils.load_one_midi(filename, self.MIDI_PATH)

    def idx2char(self, input, mode):
        result = []
        if mode == 'pitch':
            for r in input:
                r_str = [self.pitch_sample[c] for c in np.squeeze(r)]
                result.append(r_str)
        elif mode == 'duration':
            for r in input:
                r_str = [self.duration_sample[c] for c in np.squeeze(r)]
                result.append(r_str)
        return result[0]

    def getchar2idx(self, mode):
        if mode=="pitch":
            return {c: i for i, c in enumerate(self.pitch_sample)}
        elif mode=="duration":
            return {c: i for i, c in enumerate(self.duration_sample)}

    def data2idx(self, data, char2idx, limit_length=None):
        x_data = []
        for d in data:
            train_data = np.array([char2idx[i] for i in d])
            x_data.append(train_data[:])

        return np.array(x_data)

    def song2midi(self, pitches, durations, filename):
        midiutils.melody2midi(pitches, durations, filename)

if __name__=="__main__":
    util = Util()
    print (util.pitch_sample)
