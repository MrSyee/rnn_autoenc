import music21
from music21 import *
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

now = datetime.now()
NOWTIME = now.strftime("%Y%m%d-%H%M")
DURATION_LIST = [4.0, 3.5, 3.0, 2.0, 1.75, 1.5, 1.0, 0.75, 0.5, 0.25]
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def midi2melody(filename, mapping=False):
    '''
    read to midi file and extract mapped melody data
    :param
        filename: midi file name to load
        mapping: flag for output mapping
    :return:
        song_name
        n_melody: number of melody
        pitch
        duration
    '''
    songname = filename.split('/')[-1].split('.')[0]
    song = converter.parse(filename)
    part = song.parts[0]
    part_tuples = []
    try:
        track_name = part[0].bestName()
    except AttributeError:
        track_name = 'None'
    part_tuples.append(track_name)
    melody = []
    for event in part:
        for y in event.contextSites():
            if y[0] is part:
                offset = y[1]
        if getattr(event, 'isNote', None) and event.isNote:
            melody.append([event.quarterLength, event.pitch.midi, offset])
            #print([event.quarterLength, event.pitch.midi, offset])
        if getattr(event, 'isRest', None) and event.isRest:
            melody.append([event.quarterLength, 'Rest', offset])
            #print([event.quarterLength, 'Rest', offset])
    part_tuples.append(melody)
    pitch = []
    duration = []
    if (mapping == True):
        for p in melody:
            duration.append(mapping_duration(p[0]))
            pitch.append(mapping_pitch(p[1]))
    else:
        for p in melody:
            if p[0] in DURATION_LIST:
                duration.append(p[0])
            else: # deviding tied note
                if p[0] > 4.0:
                    duration.append(4.0)
                elif p[0] > 2 and p[0] < 3:
                    duration.append(2.0)
                '''
                for i in range(2):
                    d = music21.duration.quarterConversion(p[0]).components[i].quarterLength
                    duration.append(d)
                    pitch.append(p[1])
                '''
            pitch.append(p[1])

    n_melody = len(pitch)
    #print (pitch)
    #print (duration)
    #print (max([p for p in pitch if isinstance(p,int)]))
    return songname, n_melody, pitch, duration

def mapping_duration(duration):
    '''
    mapping duration 1, 2, 3, 4, 6, 8, 12, 16
    :param duration:
        duration : 1.0, 2.0, ...
    :return:
        duration : 1, 2, 3, 4, 6, 8, 12, 16
    '''
    if duration == 4.0:
        return 16
    elif duration == 3.0:
        return 12
    elif duration == 2.0:
        return 8
    elif duration == 1.5:
        return 6
    elif duration == 1.0:
        return 4
    elif duration == 0.75:
        return 3
    elif duration == 0.5:
        return 2
    elif duration == 0.25:
        return 1
    else:
        # 2.5 2.75 4.5 5.0 6.0
        return 8
        #pass
        raise Exception("wrong duration : {}".format(duration))

def mapping_pitch(pitch):
    '''
    mapping 1(C1) ~ 36(B3)
    :param pitch:
        pitch : 24 ~ 59 of 0 ~ 127
    :return:
        mapped pitch : 1 ~ 36
    '''
    if pitch == 'Rest':
        pitch = 50
        return pitch
    elif pitch >= 24 and pitch <= 59:
        pitch -= 23
        return pitch
    else:
        raise Exception("Pitch is out of range : {}".format(pitch))

def load_filename(path):
    '''
    :param
        path: loaded directory path
    :return
        all midi file name list
    '''
    file_list = [f for f in os.listdir(path) if f.endswith('.mid')]
    file_list.sort()
    return file_list

def load_all_midi(file_list, midi_path):
    '''
    :param
        file_list: midi file name list
        midi_path: loaded directory path
    :return
        all_song: list[dict{name, length, pitches, durations}), ...]
    '''
    all_song = []
    for f in file_list:
        songpath = midi_path + f
        n, l, p, d = midi2melody(songpath, mapping=False)
        song_dict = {'name': n, 'length': l, 'pitches': p, 'durations': d}
        all_song.append(song_dict)
    return all_song

def load_one_midi(filename, midi_path):
    '''
    :param
        filename: midi file name
        midi_path: loaded directory path
    :return
        song: dict{name, length, pitches, durations})
    '''
    songpath = midi_path + filename
    n, l, p, d = midi2melody(songpath, mapping=False)
    song_dict = {'name': n, 'length': l, 'pitches': p, 'durations': d}
    return song_dict

def graph_pitches(file_path, file_list):
    # draw graph about pitches of all [file_list]
    pitches = []
    for f in file_list:
        song_path = file_path + f
        _, _, p, _ = midi2melody(song_path, mapping=False)
        for i in range(len(p)):
            if p[i] == 'Rest':
                p[i] = 0
        pitches.append(p)
    for pitch in pitches:
        plt.scatter(range(len(pitch)), pitch)
        plt.ylim(-5,100)
    plt.grid(linestyle='-', linewidth=0.2)
    plt.show()

def graph_durations(file_path, file_list):
    # draw graph about durations of all [file_list]
    durations = []
    for f in file_list:
        song_path = file_path + f
        _, _, _, d = midi2melody(song_path, mapping=False)
        durations.append(d)
    for pitch in durations:
        plt.scatter(range(len(pitch)), pitch)
        plt.ylim(-5,5)
    plt.grid(linestyle='-', linewidth=0.2)
    plt.show()

def melody2midi(pitches, durations):
    '''
    make midi file
    :param
        pitches, durations
    '''
    pitch2chord = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    st = stream.Stream()
    melody = list(zip(pitches, durations))

    for p,d in melody:
        if p == 'Rest':
            n = note.Rest()
            n.duration.quarterLength = d
        else:
            p_index = p % 12
            octave = int(p / 12) - 1
            n = note.Note(pitch2chord[p_index] + str(octave))
            n.duration.quarterLength = d
        st.append(n)

    #st.show('text')
    mf = midi.translate.streamToMidiFile(st)
    if not os.path.isdir(FILE_PATH+'/generate'):
        os.mkdir(FILE_PATH+'/generate')
    path = FILE_PATH + '/generate/make_test_{}.mid'.format(NOWTIME)
    mf.open(path, 'wb')
    mf.write()
    mf.close()

def main():
    pass

if __name__=='__main__':
    file_path = './songs/'
    file_list = load_filename(file_path)
    song_file = 'test2.mid'
    song_path = file_path + song_file
    n, l, p, d = midi2melody(song_path, False)
    all_song = load_all_midi(file_list, file_path)
    all_length = [s.get('length') for s in all_song]
    print (all_length)
    #graph_pitches(file_path, file_list)
    #graph_durations(file_path, file_list)
