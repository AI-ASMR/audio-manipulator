require('dotenv').config();

const fft = require('fft-js').fft;
const fftUtil = require('fft-js').util;
const linear = require('everpolate').linear;
const fs = require('fs');

const WaveFile = require('wavefile').WaveFile;
const { unpackArray } = require('byte-data');
const { createCanvas, loadImage } = require('canvas');


const MAX_FREQUENCY = 26000; // 22 kHz highest frequency a human can hear. In order to make the spectrogram 128px we go pass that value.
const samplesLength = 256; // must be dividable by 2: 2^10=1024, 44100 samples/s => ~1m, 16ms
const pngWidth = 128; //px of segments

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

    // let fileIndex = 0
    // let spectrogramSegment = []
    // for (let i = 1; i < spectrogram.length; i++) {
    //     const sequence = spectrogram[i];
    //     spectrogramSegment.push(sequence)
    //     if (i && i % pngWidth == 0) {
    //         const currentFileName = fileName.split('.').join('-' + fileIndex + '.')
    //         fileIndex++
    //         drawSpectrogram(currentFileName, spectrogramSegment)
    //         spectrogramSegment = []
    //     }
    // }
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
            if (frequency > 110) value = 0
            let hue = 0; // for maximum magnitude of 150 k = 1,35
            let sat = '0%';
            let lit = (value > 100 ? 100 : value) + '%'; //100 is selected as the maximum possible magnitude

            if (value > maxValue) {
                maxValue = value
            } else if (value < minValue) {
                minValue = value
            }
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

function RGBToHSL(pixel) {
    // Make r, g, and b fractions of 1
    let [r, g, b, a] = pixel
    r /= 255;
    g /= 255;
    b /= 255;
    a /= 255;

    // Find greatest and smallest channel values
    let cmin = Math.min(r, g, b)
    let cmax = Math.max(r, g, b)
    let delta = cmax - cmin
    let h = 0
    let s = 0
    let l = 0

    // Calculate hue
    // No difference
    if (delta == 0)
        h = 0;
    // Red is max
    else if (cmax == r)
        h = ((g - b) / delta) % 6;
    // Green is max
    else if (cmax == g)
        h = (b - r) / delta + 2;
    // Blue is max
    else
        h = (r - g) / delta + 4;

    h = Math.round(h * 60);

    // Make negative hues positive behind 360°
    if (h < 0)
        h += 360;

    // Calculate lightness
    l = (cmax + cmin) / 2;

    // Calculate saturation
    s = delta == 0 ? 0 : delta / (1 - Math.abs(2 * l - 1));

    // Multiply l and s by 100
    s = +(s * 100).toFixed(1);
    l = +(l * 100).toFixed(1);

    return "hsl(" + h + "," + s + "%," + l + "%)";
}

// readPNGSpectrogram('heart-beat-137135.png')
readWav('bkvhi.wav', processWav) //Transforms .wav file into .png spectrogram 
//hsl(199, 100%, 50%)
//0, 127, 192, 255(Alpha)