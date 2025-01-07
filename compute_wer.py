import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import torch
from transformers import pipeline
from datasets import load_dataset
from jiwer import wer
from phonemizer.separator import Separator

LENGTH = 63

whisper = pipeline(
    "automatic-speech-recognition",
    "openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)

if __name__ == "__main__":
    # читаем также сохраненную оригинальную транскрибацию
    original_texts = {}
    with open("test/transcriptions.txt", "r", encoding="utf-8") as file:
        for line in file:
            if ": " in line:
                file_name, text = line.strip().split(": ", 1)
                original_texts[file_name] = text.lower()  # Приводим к нижнему регистру
    # Директория с тестовыми аудиофайлами
    test_dir = "test"
    
    wer_scores = []
    
    for i in range(LENGTH):
        audio_file = f"test/generated_audio_{i}.wav"
    
        # Транскрибируем аудио используя Whisper
        transcription = whisper(audio_file)
        predicted_text = transcription["text"].lower()
    
        # Достаем оригинальный текст из сохраненного файла
        original_text = original_texts.get(f"generated_audio_{i}.wav", "")
    
        if not original_text:
            print(f"Original text not found for {audio_file}. Skipping.")
            continue
    
        # Расчет WER
        wer_score = wer(original_text, predicted_text)
    
        # Сохранение результатов
        wer_scores.append(wer_score)
    
        # Вывод результатов для текущего файла
        print(f"File: {audio_file}")
        print(f"Original: {original_text}")
        print(f"Predicted: {predicted_text}")
        print(f"WER: {wer_score:.4f}")
        print("-" * 100)
    
    
    # Средние значения WER и PER
    average_wer = sum(wer_scores) / len(wer_scores)
    print(f"Average WER: {average_wer:.4f}")
