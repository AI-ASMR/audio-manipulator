const argparse = require('argparse');
const math = require('mathjs'); // Import the mathjs library

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
    let maximumMagnitude = Number.MIN_SAFE_INTEGER
    const stftMag = stftFull.map((row) => row.map((value) => {
        const modulus = math.abs(value) ** 2.0
        if (modulus > maximumMagnitude) maximumMagnitude = modulus
        return modulus
    } ));
    // Note that `stftMag` only contains the magnitudes and so we have lost the
    // phase information.
    const scale = 1.0 / maximumMagnitude;
    console.log('Maximum value in the magnitude spectrogram: ', 1 / scale);
    
    // Rescale to put all values in the range [0, 1].
    const scaledStftMag = stftMag.map((row) => row.map((value) => value * scale));
    
    // We now have a (magnitude only) spectrogram, `scaledStftMag` that is normalized to be within [0, 1.0].
    // In a practical use case, we would probably want to perform some processing on `scaledStftMag` here
    // which would produce a modified version that we would want to reconstruct audio from.

    /*
    figure(1);
    imshow(scaledStftMag.map((row) => row.map((value) => value ** 0.125)), {
        origin: 'lower',
        cmap: 'hot',
        aspect: 'auto',
        interpolation: 'nearest',
    });
    colorbar();
    title('Unmodified spectrogram');
    xlabel('time index');
    ylabel('frequency bin index');
    savefig('unmodified_spectrogram.png', { dpi: 150 });

    */

    // If the mel scale option is selected, apply a perceptual frequency scale.
    if (args.enable_mel_scale) {
        const minFreqHz = 70;
        const maxFreqHz = 8000;
        const melBinCount = 200;

        const linearBinCount = 1 + args.fft_size / 2;
        const filterbank = audioUtilities.makeMelFilterbank(
            minFreqHz,
            maxFreqHz,
            melBinCount,
            linearBinCount,
            args.sample_rate_hz
        );

        figure(2);
        imshow(filterbank, { origin: 'lower', cmap: 'hot', aspect: 'auto', interpolation: 'nearest' });
        colorbar();
        title('Mel scale filter bank');
        xlabel('linear frequency index');
        ylabel('mel frequency index');
        savefig('mel_scale_filterbank.png', { dpi: 150 });

        const melSpectrogram = math.multiply(filterbank, scaledStftMag);

        clf();
        figure(3);
        imshow(melSpectrogram.map((row) => row.map((value) => value ** 0.125)), {
            origin: 'lower',
            cmap: 'hot',
            aspect: 'auto',
            interpolation: 'nearest',
        });
        colorbar();
        title('Mel scale spectrogram');
        xlabel('time index');
        ylabel('mel frequency bin index');
        savefig('mel_scale_spectrogram.png', { dpi: 150 });

        const invertedMelToLinearFreqSpectrogram = math.multiply(math.transpose(filterbank), melSpectrogram);

        clf();
        figure(4);
        imshow(invertedMelToLinearFreqSpectrogram.map((row) => row.map((value) => value ** 0.125)), {
            origin: 'lower',
            cmap: 'hot',
            aspect: 'auto',
            interpolation: 'nearest',
        });
        colorbar();
        title('Linear scale spectrogram obtained from mel scale spectrogram');
        xlabel('time index');
        ylabel('frequency bin index');
        savefig('inverted_mel_to_linear_freq_spectrogram.png', { dpi: 150 });

        stftModified = math.transpose(invertedMelToLinearFreqSpectrogram);
    } else {
        stftModified = math.transpose(scaledStftMag);
    }
    
    // Optional: modify the spectrogram
    // For example, we can implement a low-pass filter by simply setting all frequency bins above
    // some threshold frequency (args.cutoff_freq) to 0.
    if (args.enable_filter) {
        // Calculate corresponding bin index.
        const cutoffBin = Math.round((args.cutoff_freq * args.fft_size) / args.sample_rate_hz);
        stftModified.forEach((row) => {
            for (let i = cutoffBin; i < row.length; i++) {
                row[i] = 0;
            }
        });
    }
    
    // Undo the rescaling.
    const stftModifiedScaled = stftModified.map((row) => row.map((value) => value / scale));
    const stftModifiedScaledRoot = stftModifiedScaled.map((row) => row.map((value) => value ** 0.5));
    
    // Use the Griffin&Lim algorithm to reconstruct an audio signal from the
    // magnitude spectrogram.
    let xReconstruct = audioUtilities.reconstructSignalGriffinLim(
        stftModifiedScaledRoot,
        args.fft_size,
        hopsamp,
        args.iterations
    );
        
    // The output signal must be in the range [-1, 1], otherwise we need to clip or normalize.
    let maxSample = Number.MAX_SAFE_INTEGER
    xReconstruct.forEach((val) => {
        if(Math.abs(val) > maxSample) maxSample = Math.abs(val)
    });
    if (maxSample > 1.0) {
        xReconstruct = xReconstruct.map((val) => val / maxSample);
    }

    // Save the reconstructed signal to a WAV file.
    audioUtilities.saveAudioToFile(xReconstruct, args.sample_rate_hz);

    /*
    // Save the spectrogram image also.
    clf();
    figure(5);
    imshow(stftModifiedScaledRoot.map((row) => row.map((value) => value ** 0.125)), {
        origin: 'lower',
        cmap: 'hot',
        aspect: 'auto',
        interpolation: 'nearest',
    });
    colorbar();
    title('Spectrogram used to reconstruct audio');
    xlabel('time index');
    ylabel('frequency bin index');
    savefig('reconstruction_spectrogram.png', { dpi: 150 });
    */
}

runDemo();
