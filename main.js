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
 * Addition RNN example.
 *
 * Based on Python Keras example:
 *   https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py
 */

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {createAndCompileModel, trainModel} from './train';

class AdditionRNNDemo {
    constructor(rnnType, embeddingDims, lstmUnits) {
        // Prepare training data.
        this.model = createAndCompileModel(rnnType, embeddingDims, lstmUnits);
    }

    async train(iterations, batchSize, numTestExamples, trainingSplit, validationSplit) {
        const lossValues = [
            [],
            []
        ];
        const accuracyValues = [
            [],
            []
        ];
        for (let i = 0; i < iterations; ++i) {
            const beginMs = performance.now();
            const {history, tests} = await trainModel(this.model, 1, batchSize, numTestExamples, trainingSplit, validationSplit);

            const elapsedMs = performance.now() - beginMs;
            const modelFitTime = elapsedMs / 1000;

            const trainLoss = history.history['loss'][0];
            const trainAccuracy = history.history['acc'][0];
            const valLoss = history.history['val_loss'][0];
            const valAccuracy = history.history['val_acc'][0];

            lossValues[0].push({
                'x': i,
                'y': trainLoss
            });
            lossValues[1].push({
                'x': i,
                'y': valLoss
            });

            accuracyValues[0].push({
                'x': i,
                'y': trainAccuracy
            });
            accuracyValues[1].push({
                'x': i,
                'y': valAccuracy
            });

            document.getElementById('trainStatus').textContent =
                `Iteration ${i + 1} of ${iterations}: ` +
                `Time per iteration: ${modelFitTime.toFixed(3)} (seconds)`;
            const lossContainer = document.getElementById('lossChart');
            tfvis.render.linechart(
                lossContainer, {
                    values: lossValues,
                    series: ['train', 'validation']
                }, {
                    width: 420,
                    height: 300,
                    xLabel: 'epoch',
                    yLabel: 'loss'
                });

            const accuracyContainer = document.getElementById('accuracyChart');
            tfvis.render.linechart(
                accuracyContainer, {
                    values: accuracyValues,
                    series: ['train', 'validation']
                }, {
                    width: 420,
                    height: 300,
                    xLabel: 'epoch',
                    yLabel: 'accuracy'
                });

            const examples = [];
            const isCorrect = [];
            tf.tidy(() => {
                for (let k = 0; k < tests.length; ++k) {
                    examples.push(tests[k].inputStr + ' = ' + tests[k].outputStr);
                    isCorrect.push(tests[k].inputStr.trim() === tests[k].correctAnswer.trim());
                }
            });

            const examplesDiv = document.getElementById('testExamples');
            const examplesContent = examples.map(
                (example, i) => `<div class="${isCorrect[i] ? 'answer-correct' : 'answer-wrong'}">${example}</div>`);

            examplesDiv.innerHTML = examplesContent.join('\n');
        }
    }
}

async function runAdditionRNNDemo() {
    document.getElementById('trainModel').addEventListener('click', async() => {
        const trainingSplit = +(document.getElementById('trainingSplit')).value;
        const validationSplit = +(document.getElementById('validationSplit')).value;
        const rnnTypeSelect = document.getElementById('rnnType');
        const rnnType = rnnTypeSelect.options[rnnTypeSelect.selectedIndex].getAttribute('value');
        const embeddingDims = +(document.getElementById('embeddingDims')).value;
        const lstmUnits = +(document.getElementById('lstmUnits')).value;
        const batchSize = +(document.getElementById('batchSize')).value;
        const trainIterations = +(document.getElementById('trainIterations')).value;
        const numTestExamples = +(document.getElementById('numTestExamples')).value;

        const demo = new AdditionRNNDemo(rnnType, embeddingDims, lstmUnits);
        await demo.train(trainIterations, batchSize, numTestExamples, trainingSplit, validationSplit);
    });
}

runAdditionRNNDemo();
