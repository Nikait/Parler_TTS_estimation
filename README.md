# General Plan

## 1. Run Parler-TTS on a Test Example (Example in the `results.ipynb` notebook)

**Note:** Initially, it is unclear which part of the dataset was used for the pretraining available on Hugging Face. Therefore, we will also compare on **LibriSpeech Test Clean** for objectivity.

- Split the **Jenny** dataset into training and test sets: allocate ~0.3% for the test set (resulting in 63 random audio samples from the dataset).
- Take a small portion of the **LibriSpeech Test Clean** dataset (63 random audio samples). (All steps are done in the `results.ipynb` notebook.)
- All calculations are performed on a single T4 GPU (hence, we use a small number of samples for benchmarks).

## 2. What is Included in Speech Synthesis Evaluation

We need to consider the following characteristics:

- **Naturalness**: How natural the synthesized speech sounds.
- **Intelligibility**: How clearly the words are pronounced.
- **Speaker Similarity**: How similar the synthesized voice is to the target voice.

## 3. Metrics for Evaluation

For each characteristic, we will use the following metrics:

- **Naturalness**: 
  - **MOS (Mean Opinion Score)**: A subjective score from 1 to 5, where 5 is maximum naturalness (usually evaluated by humans, which is highly subjective).
  - **Neural MOS**: For example, [UTMOS](https://arxiv.org/abs/2204.02152). We will use this metric.
  
- **Intelligibility**: 
  - **WER (Word Error Rate)**: The percentage of word recognition errors. We will use a good ASR model to transcribe the generated audio and compare it with the original transcription.

- **Speaker Similarity**: 
  - **Speaker Encoder Cosine Similarity (SECS)**: Using the WavLM model for embeddings, as described in [this paper](https://arxiv.org/abs/2104.05557).

**Why these metrics?** They are the most objective for TTS, although there are still questions: which ASR to use for WER, which model is best for Neural MOS, etc. Due to this, there may be variations in evaluations across publications, and absolute objectivity may not be achieved.

We could also evaluate more subjective aspects: manually or by asking a group of people to provide MOS scores, or assess how accurately the tone style in the Parler TTS prompt is conveyed. However, these methods are less precise.

## 4. Metric Calculations

All calculation code, including the loading of both datasets, is provided in the `results.ipynb` notebook.

Using the scripts from the repository to calculate each metric, the data must first be placed in the `./TEST_DIR` directory in the following format:

- `original_audio_{i}.wav`
- `generated_audio_{i}.wav`
- `transcriptions.txt`: Transcriptions of the files, with entries in the format `generated_audio_{i}.wav: {transcription text}`

## Results

## Metric Evaluation

| Dataset \ Metrics           | WER    | UTMOS  | SECS   |
|-----------------------------|--------|--------|--------|
| **LibriSpeech Test Clean**  | 0.4946 | 4.1036 | 0.6337 |
| **Jenny**                   | 0.2074 | 4.1682 | 0.9356 |

The benchmarks on the **LibriSpeech Test Clean** dataset performed worse. This could be due to:
1. The selected examples from the **Jenny** dataset potentially being used for training.
2. The prompt used for generating audio from the **LibriSpeech Test Clean** dataset may not have been highly relevant.
