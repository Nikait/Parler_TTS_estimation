import os
import torch
import librosa

LENGTH = 63

if __name__ == "__main__":
    # Загрузка UTMOS
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    
    # Директория с аудиофайлами
    test_dir = "test"
    
    utmos_scores = []
    
    for i in range(LENGTH):
        audio_file = os.path.join(test_dir, f"generated_audio_{i}.wav")
    
        # Загрузка аудио
        wave, sr = librosa.load(audio_file, sr=None, mono=True)
    
        # Предсказание UTMOS
        score = predictor(torch.from_numpy(wave).unsqueeze(0), sr)
    
        # Сохранение результата
        utmos_scores.append(score.item())
    
        # Вывод результата для текущего файла
        print(f"File: {audio_file}")
        print(f"UTMOS: {score.item():.4f}")
        print("-" * 100)
    
    # Среднее значение UTMOS
    average_utmos = sum(utmos_scores) / len(utmos_scores)
    print(f"Average UTMOS: {average_utmos:.4f}")
