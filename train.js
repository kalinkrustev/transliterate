/* eslint no-console:0 */
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
 * Training an attention LSTM sequence-to-sequence decoder to translate
 * various date formats into the ISO date format.
 *
 * Inspired by and loosely based on
 * https://github.com/wanasit/katakana/blob/master/notebooks/Attention-based%20Sequence-to-Sequence%20in%20Keras.ipynb
 */

const tf = require('@tensorflow/tfjs');
const translit = require('./translit');
const {createModel, runSeq2SeqInference} = require('./model');

/**
 * Generate sets of data for training.
 *
 * @param {number} trainSplit Trainining split. Must be >0 and <1.
 * @param {number} valSplit Validatoin split. Must be >0 and <1.
 * @return An `Object` consisting of
 *   - trainEncoderInput, as a `tf.Tensor` of shape
 *     `[numTrainExapmles, inputLength]`
 *   - trainDecoderInput, as a `tf.Tensor` of shape
 *     `[numTrainExapmles, outputLength]`. The first element of every
 *     example has been set as START_CODE (the sequence-start symbol).
 *   - trainDecoderOuptut, as a one-hot encoded `tf.Tensor` of shape
 *     `[numTrainExamples, outputLength, outputVocabSize]`.
 *   - valEncoderInput, same as trainEncoderInput, but for the validation set.
 *   - valDecoderInput, same as trainDecoderInput, but for the validation set.
 *   - valDecoderOutput, same as trainDecoderOuptut, but for the validation
 *     set.
 *   - testDateTuples, date tuples ([year, month, day]) for the test set.
 */
export function generateDataForTraining(trainSplit = 0.85, valSplit = 0.10) {
    tf.util.assert(
        trainSplit > 0 && valSplit > 0 && trainSplit + valSplit <= 1,
        `Invalid trainSplit (${trainSplit}) and valSplit (${valSplit})`);

    const words = require('./bg');
    tf.util.shuffle(words);

    const numTrain = Math.floor(words.length * trainSplit);
    const numVal = Math.floor(words.length * valSplit);
    console.log(`Number of words used for training: ${numTrain}`);
    console.log(`Number of words used for validation: ${numVal}`);
    console.log(`Number of words used for testing: ${words.length - numTrain - numVal}`);

    function dateTuplesToTensor(dateTuples) {
        return tf.tidy(() => {
            const inputs = translit.INPUT_FNS.map(fn => dateTuples.map(tuple => fn(tuple)));
            const inputStrings = [];
            inputs.forEach(inputs => inputStrings.push(...inputs));
            const encoderInput = translit.encodeInputStrings(inputStrings);
            const trainTargetStrings = dateTuples.map(tuple => tuple);
            let decoderInput = translit.encodeOutputStrings(trainTargetStrings).asType('float32');
            // One-step time shift: The decoder input is shifted to the left by
            // one time step with respect to the encoder input. This accounts for
            // the step-by-step decoding that happens during inference time.
            decoderInput = tf.concat([
                tf.ones([decoderInput.shape[0], 1]).mul(translit.START_CODE),
                decoderInput.slice(
                    [0, 0], [decoderInput.shape[0], decoderInput.shape[1] - 1])
            ], 1).tile([translit.INPUT_FNS.length, 1]);
            const decoderOutput = tf.oneHot(
                translit.encodeOutputStrings(trainTargetStrings),
                translit.OUTPUT_VOCAB.length).tile(
                [translit.INPUT_FNS.length, 1, 1]);
            return {
                encoderInput,
                decoderInput,
                decoderOutput
            };
        });
    }

    const {
        encoderInput: trainEncoderInput,
        decoderInput: trainDecoderInput,
        decoderOutput: trainDecoderOutput
    } = dateTuplesToTensor(words.slice(0, numTrain));
    const {
        encoderInput: valEncoderInput,
        decoderInput: valDecoderInput,
        decoderOutput: valDecoderOutput
    } = dateTuplesToTensor(words.slice(numTrain, numTrain + numVal));
    const testText =
        words.slice(numTrain + numVal, words.length);
    return {
        trainEncoderInput,
        trainDecoderInput,
        trainDecoderOutput,
        valEncoderInput,
        valDecoderInput,
        valDecoderOutput,
        testText
    };
}

async function run({epochs, batchSize, logDir, logUpdateFreq, savePath}) {
    let tfn = require('@tensorflow/tfjs-node');

    const model = createModel(
        translit.INPUT_VOCAB.length, translit.OUTPUT_VOCAB.length,
        translit.INPUT_LENGTH, translit.OUTPUT_LENGTH);
    model.summary();

    const {
        trainEncoderInput,
        trainDecoderInput,
        trainDecoderOutput,
        valEncoderInput,
        valDecoderInput,
        valDecoderOutput,
        testText
    } = generateDataForTraining();

    await model.fit(
        [trainEncoderInput, trainDecoderInput], trainDecoderOutput, {
            epochs,
            batchSize,
            shuffle: true,
            validationData: [
                [valEncoderInput, valDecoderInput], valDecoderOutput
            ],
            callbacks: {
                onEpochEnd: async(epoch, log) => {
                    await model.save(`file://${savePath}`);
                }
            }
        });

    // Save the model.
    if (savePath != null && savePath.length) {
        const saveURL = `file://${savePath}`;
        await model.save(saveURL);
        console.log(`Saved model to ${saveURL}`);
    }

    // Run seq2seq inference tests and print the results to console.
    const numTests = 20;
    for (let n = 0; n < numTests; ++n) {
        for (const testInputFn of translit.INPUT_FNS) {
            const inputStr = testInputFn(testText[n]);
            console.log('\n-----------------------');
            console.log(`Input string: ${inputStr}`);
            const correctAnswer = (testText[n]);
            console.log(`Correct answer: ${correctAnswer}`);

            const {
                outputStr
            } = await runSeq2SeqInference(model, inputStr);
            const isCorrect = outputStr === correctAnswer;
            console.log(
                `Model output: ${outputStr} (${isCorrect ? 'OK' : 'WRONG'})`);
        }
    }
}

if (require.main === module) {
    run({
        epochs: 360,
        batchSize: 20,
        savePath: './dist/model',
        logUpdateFreq: 'epoch'
    });
}

export function createAndCompileModel(rnnType, embeddingDims, lstmUnits) {
    const model = createModel(
        translit.INPUT_VOCAB.length,
        translit.OUTPUT_VOCAB.length,
        translit.INPUT_LENGTH,
        translit.OUTPUT_LENGTH,
        embeddingDims,
        lstmUnits
    );
    model.summary();
    return model;
}

export async function trainModel(model, epochs, batchSize, numTests, trainSplit, valSplit) {
    const {
        trainEncoderInput,
        trainDecoderInput,
        trainDecoderOutput,
        valEncoderInput,
        valDecoderInput,
        valDecoderOutput,
        testText
    } = generateDataForTraining(trainSplit, valSplit);

    const history = await model.fit(
        [trainEncoderInput, trainDecoderInput], trainDecoderOutput, {
            epochs,
            batchSize,
            shuffle: true,
            validationData: [
                [valEncoderInput, valDecoderInput], valDecoderOutput
            ],
            yieldEvery: 'batch'
        });

    const tests = [];
    for (let n = 0; n < numTests; ++n) {
        for (const testInputFn of translit.INPUT_FNS) {
            const inputStr = testInputFn(testText[n]);
            const correctAnswer = (testText[n]);
            const {outputStr} = await runSeq2SeqInference(model, inputStr);
            tests.push({inputStr, correctAnswer, outputStr});
        }
    }

    return {history, tests};
};
