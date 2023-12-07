require('dotenv').config();

const fft = require('fft-js').fft;
const fftUtil = require('fft-js').util;
const linear = require('everpolate').linear;
const fs = require('fs');

const WaveFile = require('wavefile').WaveFile;
const { unpackArray } = require('byte-data');
const { createCanvas } = require('canvas');


const MAX_FREQUENCY = 22000; // 22 kHz highest frequency a human can hear
const samplesLength = 512; // must be dividable by 2: 2^10=1024, 44100 samples/s => ~1m, 16ms

/**
 * FFT analysis on given samples
 * @param samples
 * @return {{magnitudes, frequencies}|{joined: Uint8Array | BigInt64Array | {magnitude: *, frequency: any}[] | Float64Array | Int8Array | Float32Array | Int32Array | Uint32Array | Uint8ClampedArray | BigUint64Array | Int16Array | Uint16Array, magnitudes, frequencies}}
 */
function analyzeSamples(samples) {
    const phasers = fft(samples);

    const frequencies = fftUtil.fftFreq(phasers, 44100); // Sample rate and coef is just used for length, and frequency step
    const magnitudes = fftUtil.fftMag(phasers);

    return { frequencies, magnitudes }
};

/**
 * Create a specter and populate it with the frequencies and magnitudes for the given samples
 * @param samples
 * @return Interpolated specter
 */
function spectro(samples) {
    const C0 = 16.35; // lowest piano note
    const NEXT_NOTE_MULTIPLIER = Math.pow(2, 1 / 12); // notes 16,25 = C0, C0 * 2^(1/12) = C# ...
    const { frequencies, magnitudes } = analyzeSamples(samples);
    let spectro = [];
    for (let i = C0; i <= MAX_FREQUENCY; i = i * NEXT_NOTE_MULTIPLIER) {
        spectro.push(i)
    }
    spectro = linear(spectro, frequencies, magnitudes);
    return spectro;
};

/**
 * Returns the number of sample sets at a sample size
 * @param wav
 * @return {number}
 */
const getNumberOfSamples = (wav) => {
    return Math.floor(wav.data.samples.length / (samplesLength * (wav.f.h / 8)));
};

/**
 * Return the sample at a given index.
 * @param {any} wav: the wave file
 * @param {number} startIndex The sample start index.
 * @param {number} stopIndex The sample stop index.
 * @return {number} The sample.
 * @throws {Error} If the sample index is off range.
 */
const getSamples = (wav, startIndex, stopIndex) => {
    startIndex = startIndex * (wav.f.h / 8);
    stopIndex = stopIndex * (wav.f.h / 8);
    if (stopIndex + wav.f.h / 8 > wav.data.samples.length) {
        const errMsg = `Range error, stopIndex ${stopIndex}, stopIndex + wav.dataType.bits ${stopIndex + wav.f.h}, wav.data.samples.length ${wav.data.samples.length}`;
        throw new Error(errMsg);
    }
    return unpackArray(
        wav.data.samples.slice(startIndex, stopIndex),
        { bits: wav.f.h, fp: wav.f.R, signed: wav.f.O, be: wav.f.o } // TODO MAKE SURE FILE DATA TYPE IS CORRECT!
    );
};

const readWav = (file, callback) => {
    let wav;
    // read the wav file
    const filePath = './audio-files/' + file;
    fs.readFile(filePath, (err, buffer) => {
        if (err) {
            return callback(err);
        }
        wav = new WaveFile(buffer);
        wav.toBitDepth("32f"); // convert to 32f for fft

        return callback(file, wav)
    })
};

const processWav = (fileName, wav, options = {}) => {

    let index = 0;
    let maxIndex;

    if (options.maxSamples) {
        maxIndex = options.maxSamples
    } else {
        maxIndex = getNumberOfSamples(wav) - 1;
    }
    const spectrogram = [];

    do {
        const samples = getSamples(wav, index * samplesLength, (index + 1) * samplesLength);
        index++;
        const snapshot = spectro(samples);
        spectrogram.push(snapshot);
    } while (index < maxIndex);

    drawSpectrogram(fileName, spectrogram)
};

function drawSpectrogram(fileName, spectrograph) {
    const strokeHeight = 1;
    const canvasHeight = spectrograph[0].length * strokeHeight;
    const canvasWidth = spectrograph.length;
    const canvas = createCanvas(canvasWidth, canvasHeight);
    const ctx = canvas.getContext('2d');
    // init canvas
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    let maxValue = 0
    let minValue = 100000000000

    spectrograph.forEach((sequence, timeSeq) => {
        sequence.forEach((value, frequency) => {
            let hue = Math.round((value * -1) + 200) < 0 ? 0 : Math.round((value * -1.35) + 200); // for maximum magnitude of 150 k = 1,35
            let sat = '100%';
            let lit = '50%';
            
            if (value > maxValue) {
                maxValue = value
            }else if (value < minValue) {
                minValue = value
            }
            ctx.beginPath();
            ctx.strokeStyle = `hsl(${hue}, ${sat}, ${lit})`;
            ctx.moveTo(timeSeq, canvasHeight - (frequency * strokeHeight));
            ctx.lineTo(timeSeq, canvasHeight - (frequency * strokeHeight + strokeHeight));
            ctx.stroke();
        });
    });
    console.log(maxValue, minValue);
    const outPath = './audio-files/' + fileName.replace('.wav', '.png')
    const out = fs.createWriteStream(outPath);
    const stream = canvas.createPNGStream();
    stream.pipe(out);
    out.on('finish', (err) => {
        if (err) { return callback(err); }
        console.log('The PNG file was created.')
    });
};


readWav('heart-beat-137135.wav', processWav)