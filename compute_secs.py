import torch
import os
import numpy as np
import librosa
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

LENGTH = 63

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "microsoft/wavlm-base-plus-sv"
)
sv_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")


def SECS(ref, gen):
    spk1_wav, _ = librosa.load(ref, sr=48000)
    spk2_wav, _ = librosa.load(gen, sr=48000)

    input1 = feature_extractor(
        [spk1_wav], padding=True, return_tensors="pt", sampling_rate=48000
    )
    if torch.cuda.is_available():
        for key in input1.keys():
            input1[key] = input1[key].to(sv_model.device)

    with torch.no_grad():
        embds_1 = sv_model(**input1).embeddings
        embds_1 = embds_1[0]

    input2 = feature_extractor(
        [spk2_wav], padding=True, return_tensors="pt", sampling_rate=48000
    )
    if torch.cuda.is_available():
        for key in input2.keys():
            input2[key] = input2[key].to(sv_model.device)

    with torch.no_grad():
        embds_2 = sv_model(**input2).embeddings
        embds_2 = embds_2[0]
    cos_sim = F.cosine_similarity(embds_1, embds_2, dim=-1).detach().cpu().numpy()

    return cos_sim



if __name__ == "__main__":
    # Директория с аудиофайлами
    test_dir = "test"
    
    # Список для хранения результатов SECS
    secs_scores = []
    
    # Пройдемся по всем файлам
    for i in range(LENGTH):
        original_audio_file = os.path.join(test_dir, f"original_audio_{i}.wav")
        generated_audio_file = os.path.join(test_dir, f"generated_audio_{i}.wav")
        
        # Проверка существования файлов
        if not os.path.exists(original_audio_file) or not os.path.exists(generated_audio_file):
            print(f"Files for index {i} not found. Skipping.")
            continue
        
        # Расчет SECS
        secs_score = SECS(original_audio_file, generated_audio_file)
        secs_scores.append(secs_score)
        
        # Вывод результата для текущего файла
        print(f"File: {i}")
        print(f"Original: {original_audio_file}")
        print(f"Generated: {generated_audio_file}")
        print(f"SECS: {secs_score:.4f}")
        print("-" * 50)
        
    # Среднее значение SECS
    average_secs = np.mean(secs_scores)
    print(f"Average SECS: {average_secs:.4f}")
