const argparse = require('argparse');
const tf = require('@tensorflow/tfjs-node');

const audioUtilities = require('./audio-utils'); // Replace with the correct path

function runDemo() {
    const parser = new argparse.ArgumentParser();
    parser.add_argument('--in_file', { type: String, default: 'bkvhi.wav', help: 'Input WAV file' });
    parser.add_argument('--sample_rate_hz', { type: Number, default: 44100, help: 'Sample rate in Hz' });
    parser.add_argument('--fft_size', { type: Number, default: 2048, help: 'FFT size' });
    parser.add_argument('--iterations', { type: Number, default: 300, help: 'Number of iterations to run' });
    parser.add_argument('--enable_filter', { action: 'store_true', help: 'Apply a low-pass filter' });
    parser.add_argument('--enable_mel_scale', { action: 'store_true', help: 'Convert to mel scale and back' });
    parser.add_argument('--cutoff_freq', { type: Number, default: 1000, help: 'If filter is enabled, the low-pass cutoff frequency in Hz' });

    const args = parser.parse_args();

    const in_file = args.in_file;

    let stftModified;

    // Load an audio file. It must be WAV format. Multi-channel files will be
    // converted to mono.
    const input_signal = audioUtilities.getSignal(in_file, args.sample_rate_hz);
    // Hopsamp is the number of samples that the analysis window is shifted after
    // computing the FFT. For example, if the sample rate is 44100 Hz and hopsamp is
    // 256, then there will be approximately 44100/256 = 172 FFTs computed per second
    // and thus 172 spectral slices (i.e., columns) per second in the spectrogram.
    const hopsamp = args.fft_size / 8;
    // Compute the Short-Time Fourier Transform (STFT) from the audio file.
    const stftFull = audioUtilities.stftForReconstruction(input_signal, args.fft_size, hopsamp);

    // Note that the STFT is complex-valued. Therefore, to get the (magnitude)
    // spectrogram, we need to take the absolute value.
    const stftMag = stftFull.abs().square()
    const maximumMagnitude = stftMag.max()
    // Note that `stftMag` only contains the magnitudes and so we have lost the
    // phase information.
    const scale = maximumMagnitude.reciprocal();
    console.log('Maximum value in the magnitude spectrogram: ');
    maximumMagnitude.print()

    // Rescale to put all values in the range [0, 1].
    const scaledStftMag = stftMag.mul(scale)

    stftModified = scaledStftMag;

    // Undo the rescaling.
    const stftModifiedScaled = stftModified.mul(maximumMagnitude);
    const stftModifiedScaledRoot = stftModifiedScaled.sqrt();
    // console.log(tf.util.sizeFromShape(stftModifiedScaledRoot))

    // Use the Griffin&Lim algorithm to reconstruct an audio signal from the
    // magnitude spectrogram.
    let reconstructTensor = audioUtilities.reconstructSignalGriffinLim(
        stftModifiedScaledRoot,
        args.fft_size,
        hopsamp,
        args.iterations
    );

    // The output signal must be in the range [-1, 1], otherwise we need to clip or normalize.
    let maxSample = reconstructTensor.abs().max()
    if (maxSample > 1.0) {
        reconstructTensor = reconstructTensor.mul(maxSample);
    }
    // Save the reconstructed signal to a WAV file.
    audioUtilities.saveAudioToFile(reconstructTensor, args.sample_rate_hz);

}

runDemo();
