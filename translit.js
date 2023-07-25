/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * Date formats and conversion utility functions.
 *
 * This file is used for the training of the date-conversion model and
 * date conversions based on the trained model.
 *
 * It contains functions that generate random dates and represent them in
 * several different formats such as (2019-01-20 and 20JAN19).
 * It also contains functions that convert the text representation of
 * the dates into one-hot `tf.Tensor` representations.
 */

const tf = require('@tensorflow/tfjs');

const INPUT_LENGTH = 21;
const OUTPUT_LENGTH = 18;

// Use "\n" for padding for both input and output. It has to be at the
// beginning so that `mask_zero=True` can be used in the keras model.
const INPUT_VOCAB = '\nabcdefghijklmnopqrstuvwxyz';

// OUTPUT_VOCAB includes an start-of-sequence (SOS) token, represented as
// '\t'. Note that the date strings are represented in terms of their
// constituent characters, not words or anything else.
const OUTPUT_VOCAB = '\n\tабвгдежзийклмнопрстуфхцчшщъьюя';

const START_CODE = 1;

const dict = {
    'а': 'a',
    'б': 'b',
    'в': 'v',
    'г': 'g',
    'д': 'd',
    'е': 'e',
    'ж': 'zh',
    'з': 'z',
    'и': 'i',
    'й': 'y',
    'к': 'k',
    'л': 'l',
    'м': 'm',
    'н': 'n',
    'о': 'o',
    'п': 'p',
    'р': 'r',
    'с': 's',
    'т': 't',
    'у': 'u',
    'ф': 'f',
    'х': 'h',
    'ц': 'ts',
    'ч': 'ch',
    'ш': 'sh',
    'щ': 'sht',
    'ъ': 'a',
    'ь': 'y',
    'ю': 'yu',
    'я': 'ya'
};

function transliterate(text) {
    return text.split('').map(char => dict[char] || char).join('');
}

const INPUT_FNS = [
    transliterate
]; // TODO(cais): Add more formats if necessary.

/**
 * Encode a number of input date strings as a `tf.Tensor`.
 *
 * The encoding is a sequence of one-hot vectors. The sequence is
 * padded at the end to the maximum possible length of any valid
 * input date strings. The padding value is zero.
 *
 * @param {string[]} dateStrings Input date strings. Each element of the array
 *   must be one of the formats listed above. It is okay to mix multiple formats
 *   in the array.
 * @returns {tf.Tensor} One-hot encoded characters as a `tf.Tensor`, of dtype
 *   `float32` and shape `[numExamples, maxInputLength]`, where `maxInputLength`
 *   is the maximum possible input length of all valid input date-string formats.
 */
function encodeInputStrings(dateStrings) {
    const n = dateStrings.length;
    const x = tf.buffer([n, INPUT_LENGTH], 'float32');
    for (let i = 0; i < n; ++i) {
        for (let j = 0; j < INPUT_LENGTH; ++j) {
            if (j < dateStrings[i].length) {
                const char = dateStrings[i][j];
                const index = INPUT_VOCAB.indexOf(char);
                if (index === -1) {
                    throw new Error(`Unknown char: ${char}`);
                }
                x.set(index, i, j);
            }
        }
    }
    return x.toTensor();
}

/**
 * Encode a number of output date strings as a `tf.Tensor`.
 *
 * The encoding is a sequence of integer indices.
 *
 * @param {string[]} dateStrings An array of output date strings, must be in the
 *   ISO date format (YYYY-MM-DD).
 * @returns {tf.Tensor} Integer indices of the characters as a `tf.Tensor`, of
 *   dtype `int32` and shape `[numExamples, outputLength]`, where `outputLength`
 *   is the length of the standard output format (i.e., `10`).
 */
function encodeOutputStrings(dateStrings, oneHot = false) {
    const n = dateStrings.length;
    const x = tf.buffer([n, OUTPUT_LENGTH], 'int32');
    for (let i = 0; i < n; ++i) {
        for (let j = 0; j < OUTPUT_LENGTH; ++j) {
            if (j < dateStrings[i].length) {
                const char = dateStrings[i][j];
                const index = OUTPUT_VOCAB.indexOf(char);
                if (index === -1) {
                    throw new Error(`Unknown char: ${char}`);
                }
                x.set(index, i, j);
            }
        }
    }
    return x.toTensor();
}

module.exports = {
    INPUT_LENGTH,
    OUTPUT_LENGTH,
    INPUT_VOCAB,
    OUTPUT_VOCAB,
    START_CODE,
    INPUT_FNS,
    encodeInputStrings,
    encodeOutputStrings
};
