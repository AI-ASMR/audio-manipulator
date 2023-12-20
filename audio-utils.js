const fs = require('fs');
const math = require('mathjs');
const fft = require('fft-js').fft;
const WaveFile = require('wavefile').WaveFile;

function hzToMel(fHz) {
    return 2595 * math.log10(1.0 + fHz / 700.0);
}

function melToHz(mMel) {
    return 700 * (math.pow(10, mMel / 2595) - 1.0);
}

function fftBinToHz(nBin, sampleRateHz, fftSize) {
    return (nBin * sampleRateHz) / (2.0 * fftSize);
}

function hzToFFTBin(fHz, sampleRateHz, fftSize) {
    return Math.round((fHz * 2.0 * fftSize) / sampleRateHz);
}

function hanningWindow(length) {
    const window = [];
    for (let i = 0; i < length; i++) {
        window.push(0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1))));
    }
    return window;
}

function makeMelFilterbank(minFreqHz, maxFreqHz, melBinCount, linearBinCount, sampleRateHz) {
    const minMels = hzToMel(minFreqHz);
    const maxMels = hzToMel(maxFreqHz);
    const melLinSpaced = math.linspace(minMels, maxMels, melBinCount);
    const centerFrequenciesHz = melLinSpaced.map((mel) => melToHz(mel));
    const melsPerBin = (maxMels - minMels) / (melBinCount - 1);
    const melsStart = minMels - melsPerBin;
    const hzStart = melToHz(melsStart);
    const fftBinStart = hzToFFTBin(hzStart, sampleRateHz, linearBinCount);
    const melsEnd = maxMels + melsPerBin;
    const hzStop = melToHz(melsEnd);
    const fftBinStop = hzToFFTBin(hzStop, sampleRateHz, linearBinCount);

    const linearBinIndices = centerFrequenciesHz.map((fHz) => hzToFFTBin(fHz, sampleRateHz, linearBinCount));

    const filterbank = math.zeros(melBinCount, linearBinCount);

    for (let melBin = 0; melBin < melBinCount; melBin++) {
        const centerFreqLinearBin = linearBinIndices[melBin];

        if (centerFreqLinearBin > 1) {
            const leftBin = melBin === 0 ? Math.max(0, fftBinStart) : linearBinIndices[melBin - 1];

            for (let fBin = leftBin; fBin <= centerFreqLinearBin; fBin++) {
                if (centerFreqLinearBin - leftBin > 0) {
                    const response = (fBin - leftBin) / (centerFreqLinearBin - leftBin);
                    filterbank[melBin][fBin] = response;
                }
            }
        }

        if (centerFreqLinearBin < linearBinCount - 2) {
            const rightBin = melBin === melBinCount - 1 ? Math.min(linearBinCount - 1, fftBinStop) : linearBinIndices[melBin + 1];

            for (let f_bin = centerFreqLinearBin; f_bin <= rightBin; f_bin++) {
                if (rightBin - centerFreqLinearBin > 0) {
                    const response = (rightBin - f_bin) / (rightBin - centerFreqLinearBin);
                    filterbank[melBin][f_bin] = response;
                }
            }
        }

        filterbank[melBin][centerFreqLinearBin] = 1.0;
    }

    return filterbank;
}

function stftForReconstruction(x, fftSize, hopsamp) {

    // TODO NEED TO MASSIVELY OPTIMIZE THE PERFORMANCE IN THIS FUNCTION!!!
    const window = hanningWindow(fftSize);
    fftSize = parseInt(fftSize);
    hopsamp = parseInt(hopsamp);
    const stft = [];
    for (let i = 0; i < x.length - fftSize; i += hopsamp) {
        const segment = x.slice(i, i + fftSize);
        const windowedSegment = math.dotMultiply(segment, window);
        const fftResult = fft(windowedSegment);
        stft.push(fftResult);
    }
    return stft;
}

function istftForReconstruction(X, fftSize, hopsamp) {
    const window = hanningWindow(fftSize);
    fftSize = parseInt(fftSize);
    hopsamp = parseInt(hopsamp);
    const timeSlices = X.length;
    const lenSamples = timeSlices * hopsamp + fftSize;
    const x = new Array(lenSamples).fill(0);

    for (let n = 0; n < timeSlices; n++) {
        const i = n * hopsamp;
        const frame = math.ifft(X[n]);
        for (let j = 0; j < fftSize; j++) {
            x[i + j] += window[j] * (frame[j] ? frame[j].re : 0);
        }
    }

    return x;
}

function getSignal(in_file, expectedSampleRate = 44100) {
    const data = fs.readFileSync(in_file);
    const wav = new Uint8Array(data);
    const sampleRate = wav[24] + (wav[25] << 8) + (wav[26] << 16) + (wav[27] << 24);
    const num_type = wav[34] === 16 ? 'int16' : wav[34] === 32 ? 'int32' : wav[34] === 64 ? 'float64' : null;
    if (num_type === null) {
        throw new Error('Unknown format.');
    }

    let y;
    let res = []
    if (num_type === 'int16') {
        y = new Int16Array(wav.buffer.slice(44));
        y.forEach(val => res.push(val / 32768));
    } else if (num_type === 'int32') {
        y = new Int32Array(wav.buffer.slice(44));
        y.forEach(val => res.push(val / 2147483648));
    } else if (num_type === 'float64') {
        y = new Float64Array(wav.buffer.slice(44));
        res = [...y];
    }

    if (sampleRate != expectedSampleRate) {
        throw new Error('Invalid sample rate.');
    }
    return res;
}

function reconstructSignalGriffinLim(magnitudeSpectrogram, fftSize, hopsamp, iterations) {
    const timeSlices = magnitudeSpectrogram.length;
    const lenSamples = timeSlices * hopsamp + fftSize;
    let xReconstruct = math.random(lenSamples);

    let n = iterations; // number of iterations of Griffin-Lim algorithm.
    while (n > 0) {
        n--;
        const reconstructionSpectrogram = stftForReconstruction(xReconstruct, fftSize, hopsamp);
        const reconstructionAngle = reconstructionSpectrogram.map(frame =>
            frame.map(complex => (complex ? Math.atan2(complex.im, complex.re) : 0))
        );
        const proposalSpectrogram = magnitudeSpectrogram.map((frame, i) =>
            frame.map((magnitude, j) => {
                const angle = reconstructionAngle[i] ? reconstructionAngle[i][j] : 0;
                return math.multiply(magnitude, angle ? math.exp(math.complex(0, angle)) : 1);
            })
        );
        xReconstruct = istftForReconstruction(proposalSpectrogram, fftSize, hopsamp);
        console.log(`Reconstruction iteration: ${iterations - n}/${iterations}`);
    }

    return xReconstruct;
}

function saveAudioToFile(x, sampleRate, outFile = 'out.wav') {

    const xMax = math.max(math.abs(x));
    if (xMax > 1.0) {
        // Normalize the audio signal if its maximum value is greater than 1.0
        x = x.map((val) => val / xMax);
    }

    // Rescale to the range [-32767, 32767]
    x = math.multiply(x, 32767.0);

    // Create a WaveFile instance
    const wav = new WaveFile();

    // Set WAV file parameters
    wav.fromScratch(1, sampleRate, '16', x);

    // Write the WaveFile instance to a file
    fs.writeFileSync(outFile, wav.toBuffer());

    console.log(`Audio saved to ${outFile}`);
}

module.exports = {
    hzToMel,
    melToHz,
    fftBinToHz,
    hzToFFTBin,
    makeMelFilterbank,
    stftForReconstruction,
    istftForReconstruction,
    getSignal,
    reconstructSignalGriffinLim,
    saveAudioToFile,
};

/*
// Example usage:
const inFile = 'input.wav';
const outFile = 'output.wav';
const expectedSampleRate = 44100;
const fftSize = 1024;
const hopSize = 256;
const iterations = 100;

const signal = getSignal(inFile, expectedSampleRate);
const magnitudeSpectrogram = stftForReconstruction(signal, fftSize, hopSize);
const reconstructedSignal = reconstructSignalGriffinLim(magnitudeSpectrogram, fftSize, hopSize, iterations);
saveAudioToFile(reconstructedSignal, expectedSampleRate, outFile);
*/