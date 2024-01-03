const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const math = require('mathjs');
const WaveFile = require('wavefile').WaveFile;
const { createCanvas, loadImage } = require('canvas');

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
    let tensor
    if (Array.isArray(x)) {
        tensor = tf.tensor(x)
    } else {
        tensor = x
    }
    const stft = tf.signal.stft(tensor, fftSize, hopsamp)
    return stft;
}

/**
 * Invert a STFT into a time domain signal using TensorFlow.js.
 * 
 * @param {tf.Tensor} spectrogram - Input spectrogram. The rows are the time slices and columns are the frequency bins.
 * @param {number} fftSize - FFT size.
 * @param {number} hopsamp - The hop size, in samples.
 * @returns {tf.Tensor} - The inverse STFT as a TensorFlow.js tensor.
 */
function istftForReconstruction(spectrogram, fftSize, hopSamp) {
    const fftSizeInt = parseInt(fftSize);
    const hopSampInt = parseInt(hopSamp);

    const window = tf.signal.hannWindow(fftSizeInt);

    const timeSlices = spectrogram.shape[0];
    const lenSamples = parseInt(timeSlices * hopSampInt + fftSizeInt - 1);

    let x = new Array(lenSamples).fill(0)
    for (let n = 0, i = 0; i < lenSamples - fftSizeInt; n++, i += hopSampInt) {
        const values = window.mul(tf.spectral.irfft(spectrogram.slice(n, 1)))
        const valuesData = values.dataSync()
        for (let j = 0; j < fftSizeInt; j++) {
            x[i + j] += valuesData[j]
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
/**
 * 
 * @param {tf.Tensor} magnitudeSpectrogram 
 * @param {number} fftSize 
 * @param {number} hopsamp 
 * @param {number} iterations 
 * @returns {tf.Tensor}
 */
function reconstructSignalGriffinLim(magnitudeSpectrogram, fftSize, hopsamp, iterations) {
    const timeSlices = magnitudeSpectrogram.shape[0];
    const lenSamples = timeSlices * hopsamp + fftSize - 1;
    let xReconstruct = tf.randomNormal([lenSamples])

    let n = iterations; // number of iterations of Griffin-Lim algorithm.
    while (n > 0) {
        n--;
        const reconstructionSpectrogram = stftForReconstruction(xReconstruct, fftSize, hopsamp);

        const reconstructionSpectrogramData = reconstructionSpectrogram.dataSync()
        const reconstructionAngle = []
        const row = []
        const rowLength = reconstructionSpectrogram.shape[1]
        for (let i = 0; i < reconstructionSpectrogramData.length; i += 2) {
            row.push(Math.atan2(reconstructionSpectrogramData[i + 1], reconstructionSpectrogramData[i]))
            if ((i + 2) % (rowLength * 2) == 0) {
                reconstructionAngle.push([...row]);
                row.length = 0;
            }
        }
        const zeroTensor = tf.fill(magnitudeSpectrogram.shape, 0)
        const reconstructionAnglePhase = tf.complex(zeroTensor, reconstructionAngle)
        const proposalSpectrogram = magnitudeSpectrogram.mul(reconstructionAnglePhase.exp())
        const reconstructArray = istftForReconstruction(proposalSpectrogram, fftSize, hopsamp);
        xReconstruct = tf.tensor(reconstructArray)
        console.log(`Reconstruction iteration: ${iterations - n}/${iterations}`);
    }

    return xReconstruct;
}

function saveAudioToFile(x, sampleRate, outFile = 'out.wav') {

    const xMax = x.abs().max();
    if (xMax > 1.0) {
        // Normalize the audio signal if its maximum value is greater than 1.0
        x = x.div(xMax);
    }

    // Rescale to the range [-32767, 32767]
    x = x.mul(32767.0);

    // Create a WaveFile instance
    const wav = new WaveFile();

    const waveData = x.dataSync();

    // Set WAV file parameters
    wav.fromScratch(1, sampleRate, '16', waveData);

    // Write the WaveFile instance to a file
    fs.writeFileSync(outFile, wav.toBuffer());

    console.log(`Audio saved to ${outFile}`);
}

function drawSpectrogram(fileName, spectrograph) {
    const strokeHeight = 1;
    const canvasHeight = spectrograph[0].length * strokeHeight;
    const canvasWidth = spectrograph.length;
    const canvas = createCanvas(canvasWidth, canvasHeight);
    const ctx = canvas.getContext('2d');
    // init canvas
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    spectrograph.forEach((sequence, timeSeq) => {
        sequence.forEach((value, frequency) => {
            if (frequency > 110) value = 0
            let hue = 0;
            let sat = '0%';
            let lit = (value > 100 ? 100 : value) + '%'; //100 is selected as the maximum possible magnitude

            ctx.beginPath();
            ctx.strokeStyle = `hsl(${hue}, ${sat}, ${lit})`;
            ctx.moveTo(timeSeq, canvasHeight - (frequency * strokeHeight));
            ctx.lineTo(timeSeq, canvasHeight - (frequency * strokeHeight + strokeHeight));
            ctx.stroke();
        });
    });
    const outPath = './audio-files/' + fileName.replace('.wav', '.png')
    const out = fs.createWriteStream(outPath);
    const stream = canvas.createPNGStream();
    stream.pipe(out);
    out.on('finish', (err) => {
        if (err) { return callback(err); }
        console.log('The PNG file was created.')
    });
};

function readPNGSpectrogram(fileName) {
    const filePath = './audio-files/' + fileName
    loadImage(filePath)
        .then((image) => {
            const imgHeight = image.height
            const imgWidth = image.width

            const canvas = createCanvas(imgWidth, imgHeight);
            const ctx = canvas.getContext('2d');

            ctx.drawImage(image, 0, 0);

            const imageData = ctx.getImageData(0, 0, imgWidth, imgHeight).data

            const splitPixels = []

            for (let i = 0; i < imageData.length; i += 4) {
                const pixel = imageData.slice(i, i + 4);
                splitPixels.push(pixel);
            }

            const pixels = splitPixels.map(RGBToHSL)
            console.table(pixels)
        })
        .catch(err => {
            console.log('oh no!', err)
        })
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
    drawSpectrogram,
    readPNGSpectrogram
};
