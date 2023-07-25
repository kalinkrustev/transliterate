/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 * The client-side inference part of the date-conversion example.
 *
 * Based on Python Keras example:
 *   https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py
 */

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

import {
    INPUT_FNS,
    INPUT_LENGTH
} from './translit';
import {
    runSeq2SeqInference
} from './model';

const RELATIVE_MODEL_URL = './model/model-128-64.json';

const status = document.getElementById('status');
const inputDateString = document.getElementById('input-date-string');
const outputDateString = document.getElementById('output-date-string');
const attentionHeatmap = document.getElementById('attention-heatmap');
const randomButton = document.getElementById('random-date');

let model;

inputDateString.addEventListener('change', async() => {
    let inputStr = inputDateString.value.trim();

    if (inputStr.length > INPUT_LENGTH) {
        inputStr = inputStr.slice(0, INPUT_LENGTH);
    }

    try {
        const getAttention = true;
        const t0 = tf.util.now();
        const {
            outputStr,
            attention
        } =
        await runSeq2SeqInference(model, inputStr, getAttention);
        const tElapsed = tf.util.now() - t0;
        status.textContent = `seq2seq conversion took ${tElapsed.toFixed(1)} ms`;
        outputDateString.value = outputStr;

        const xTickLabels =
            (outputStr + ' '.repeat(18)).substr(0, 18).split('').map((char, i) => `(${i + 1}) "${char}"`);
        const yTickLabels = [];
        for (let i = 0; i < INPUT_LENGTH; ++i) {
            if (i < inputStr.length) {
                yTickLabels.push(`(${i + 1}) "${inputStr[i]}"`);
            } else {
                yTickLabels.push(`(${i + 1}) ""`);
            }
        }
        await tfvis.render.heatmap(
            attentionHeatmap, {
                values: attention.squeeze([0]),
                xTickLabels,
                yTickLabels
            }, {
                width: 600,
                height: 360,
                xLabel: 'Output characters',
                yLabel: 'Input characters',
                colorMap: 'blues'
            });
    } catch (err) {
        outputDateString.value = err.message;
        console.error(err);
    }
});

randomButton.addEventListener('click', async() => {
    inputDateString.value = 'dokumentaciya';
    inputDateString.dispatchEvent(new Event('change'));
});

async function init() {
    status.textContent = `Loading model from ${RELATIVE_MODEL_URL} ...`;
    model = await tf.loadLayersModel(RELATIVE_MODEL_URL);
    status.textContent = 'Done loading model.';
    model.summary();

    const exampleItems = document.getElementsByClassName('input-date-example');
    for (const exampleItem of exampleItems) {
        exampleItem.addEventListener('click', (event) => {
            inputDateString.value = event.srcElement.textContent;
            inputDateString.dispatchEvent(new Event('change'));
        });
    }

    inputDateString.dispatchEvent(new Event('change'));
}

init();
