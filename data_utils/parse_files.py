import os
import math
import scipy.io.wavfile as wav
import numpy as np
from pipes import quote
from .labelers import by_name

# Importing

def convert_wav_files_to_nptensor(directory, block_size, out_file, label=by_name):
    files = []
    for file in os.listdir(directory):
        if file.endswith('.wav') and not file.startswith('.'):
            files.append(directory+file)

    blocks_X = []
    blocks_Y = []
    for (i, f) in enumerate(files):
        X = load_training_example(f, block_size)
        print('Processing {}/{}: {} ({} blocks)'
              .format(i+1, len(files), f, len(X)))
        Y = label(X, f)
        blocks_X.extend(X)
        blocks_Y.extend(Y)

    x_data = np.array(blocks_X)
    print("yblocks type: {}, len: {}, first: {}".format(type(blocks_Y), len(blocks_Y), blocks_Y[0]))
    y_data = np.array(blocks_Y)

    blockshape = x_data.shape[1:] if len(x_data.shape) > 1 else (1)
    print("X data: {} blocks of size {}, type {}"
          .format(x_data.shape[0], blockshape, x_data.dtype))

    yblockshape = y_data.shape[1:] if len(y_data.shape) > 1 else (1)
    print("Y labels: {} blocks of size {}, type {}"
          .format(y_data.shape[0], yblockshape, y_data.dtype))

    np.save(out_file+'_x', x_data)
    np.save(out_file+'_y', y_data)
    print("Wrote to disk")


def identity_transformer(block):
    return block

def rfft_transformer(block):
    """Takes the real fft of a block and returns a block of the same size
    where A[0:n/2] are magnitudes and A[n/2:n] are angles of the frequencies.
    """

    fft_block = np.fft.rfft(block)
    return np.concatenate((np.abs(fft_block), np.angle(fft_block)))


def load_training_example(filename, block_size=2048, transformer=rfft_transformer):
    data, samplerate = read_wav_as_np(filename)
    num_samples = data.shape[0]
    num_blocks = int(math.ceil(float(num_samples) / block_size))    # y u no //?
    blocks = []

    for i in range(0, num_samples, block_size):
        block = data[i:i+block_size]
        if (block.shape[0] < block_size):       # The last block
            padding = np.zeros((block_size - block.shape[0],))
            block = np.concatenate((block, padding))
        transformed_block = transformer(block)
        blocks.append(transformed_block)

    return blocks

def read_wav_as_np(filename):
    sample_rate, data = wav.read(filename)
    if (len(data.shape) > 1):
        print("{}: multichannel audio, taking first channel only".format(filename))
        data = data[:, 0]
    np_arr = data.astype('float32') / 32768.0    # Normalize 16-bit input to [-1, 1] range
    return np_arr, sample_rate


# Exporting

def write_np_as_wav(X, sample_rate, filename):
    Xnew = X * 32767.0
    Xnew = Xnew.astype('int16')
    wav.write(filename, sample_rate, Xnew)
    return

def convert_nptensor_to_wav_files(tensor, indices, filename, useTimeDomain=False):
    num_seqs = tensor.shape[1]
    for i in indices:
        chunks = []
        for x in xrange(num_seqs):
            chunks.append(tensor[i][x])
        save_generated_example(filename+str(i)+'.wav', chunks, useTimeDomain=useTimeDomain)

def save_generated_example(filename, generated_sequence, useTimeDomain=False, sample_frequency=44100):
    if useTimeDomain:
        time_blocks = generated_sequence
    else:
        time_blocks = fft_blocks_to_time_blocks(generated_sequence)
    song = convert_sample_blocks_to_np_audio(time_blocks)
    write_np_as_wav(song, sample_frequency, filename)
    return

def convert_sample_blocks_to_np_audio(blocks):
    song_np = np.concatenate(blocks)
    return song_np

def fft_blocks_to_time_blocks(blocks_ft_domain):
    time_blocks = []
    for block in blocks_ft_domain:
        num_elems = block.shape[0] / 2
        real_chunk = block[0:num_elems]
        imag_chunk = block[num_elems:]
        new_block = real_chunk + 1.0j * imag_chunk
        time_block = np.fft.ifft(new_block)
        time_blocks.append(time_block)
    return time_blocks

#def audio_unit_test(filename, filename2):
#    data, bitrate = read_wav_as_np(filename)
#    time_blocks = convert_np_audio_to_sample_blocks(data, 1024)
#    ft_blocks = time_blocks_to_fft_blocks(time_blocks)
#    time_blocks = fft_blocks_to_time_blocks(ft_blocks)
#    song = convert_sample_blocks_to_np_audio(time_blocks)
#    write_np_as_wav(song, bitrate, filename2)
#    return

def convert_mp3_to_wav(filename, sample_frequency):
    ext = filename[-4:]
    if(ext != '.mp3'):
        return
    files = filename.split('/')
    orig_filename = files[-1][0:-4]
    orig_path = filename[0:-len(files[-1])]
    new_path = ''
    if(filename[0] == '/'):
        new_path = '/'
    for i in xrange(len(files)-1):
        new_path += files[i]+'/'
    tmp_path = new_path + 'tmp'
    new_path += 'wave'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    filename_tmp = tmp_path + '/' + orig_filename + '.mp3'
    new_name = new_path + '/' + orig_filename + '.wav'
    sample_freq_str = "{0:.1f}".format(float(sample_frequency)/1000.0)
    cmd = 'lame -a -m m {0} {1}'.format(quote(filename), quote(filename_tmp))
    os.system(cmd)
    cmd = 'lame --decode {0} {1} --resample {2}'.format(quote(filename_tmp), quote(new_name), sample_freq_str)
    os.system(cmd)
    return new_name

def convert_flac_to_wav(filename, sample_frequency):
    ext = filename[-5:]
    if(ext != '.flac'):
        return
    files = filename.split('/')
    orig_filename = files[-1][0:-5]
    orig_path = filename[0:-len(files[-1])]
    new_path = ''
    if(filename[0] == '/'):
        new_path = '/'
    for i in xrange(len(files)-1):
        new_path += files[i]+'/'
    new_path += 'wave'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    new_name = new_path + '/' + orig_filename + '.wav'
    cmd = 'sox {0} {1} channels 1 rate {2}'.format(quote(filename), quote(new_name), sample_frequency)
    os.system(cmd)
    return new_name

def convert_folder_to_wav(directory, sample_rate=44100):
    for file in os.listdir(directory):
        fullfilename = directory+file
        if file.endswith('.mp3'):
            convert_mp3_to_wav(filename=fullfilename, sample_frequency=sample_rate)
        if file.endswith('.flac'):
            convert_flac_to_wav(filename=fullfilename, sample_frequency=sample_rate)
    return directory + 'wave/'

