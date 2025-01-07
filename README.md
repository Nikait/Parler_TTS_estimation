# Общий план

## 1. Запустим Parler-TTS на тестовом примере (пример в ноутбуке results.ipynb).


Note: изначально непонятно, какая часть всей выборки использовалась для претрейна, который выложен на huggingface. Поэтому также будем сравнивать на LibriSpeech test clean для объективности.

Разделим датасет Jenny на тренировочный и тестовый наборы: выделим под тестовую часть ~0.3% (в итоге 63 случайных аудиозаписей из набора).

Также возьмем небольшую часть датасета LibriSpeech test clean (63 случайных аудиозаписей). (всё проделано в ноутбуке results.ipynb)

Все расчеты проводятся на одной T4 GPU (поэтому для бенчмарков используем немного записей).

## 2. Что входит в оценку синтеза речи

Нам важно учитывать характеристики:

- Естественность (Naturalness)

- Разборчивость (Intelligibility): Насколько четко произносятся слова.

- Сходство голосов: насколько синтезированный голос похож на целевой.

## 3. Метрики для оценки

Для каждой характеристики будем использовать следующие метрики:

- Естественность: MOS (Mean Opinion Score) — субъективная оценка от 1 до 5, где 5 — максимальная естественность (обычно оценивается людьми, что очень субьективно). Также есть разные Neural MOS, к примеру [UTMOS](https://arxiv.org/abs/2204.02152), будем использовать его.
- Разборчивость: WER (Word Error Rate) — процент ошибок в распознавании слов. Будем брать хороший ASR, полученное аудио транскрибировать и сравнивать с исходной транскрибацией.

- Сходство голосов: будем считать [Speaker Encoder Cosine Similarity - SECS](https://arxiv.org/abs/2104.05557), используя модель эмбеддингов WavLM

Почему именно эти метрики? Они являются наиболее объективными для TTS, хотя и тут вопросы: какой ASR использовать для WER, какую именно модель для Neural MOS лучше использовать, из-за этого во многих публикациях могут быть различия в оценках и не будет объективности.

Можно также было бы оценивать более субъективные вещи: вручную и/или попросить группу людей сделать оценку MOS, также насколько точно передается стиль тона в промпте Parler TTS, но это не сильно точно.


## 4. Расчеты метрик

Весь код расчета, в т.ч. загрузка обоих датасетов представлены в ноутбуке results.ipynb.

Используя скрипты из репозитория для расчета каждой отдельной метрики, перед этим данные должны быть помещены в директорию ./test/ в следующем формате:

- original_audio_{i}.wav
- generated_audio_{i}.wav
- transcriptions.txt - траскрибация файлов, записи в виде generated_audio_{i}.wav: {текст транкрибации}

## Результаты

## Оценка метрик

| Датасет \ метрики           | WER    | UTMOS  | SECS   |
|-----------------------------|--------|--------|--------|
| **LibriSpeech Test Clean**  | 0.4946 | 4.1036 | 0.6337 |
| **Jenny**                   | 0.2074 | 4.1682 | 0.9356 |


Бенчмарки на датасете **LibriSpeech Test Clean** получились хуже, что во-первых, возможно из-за того, что аудио из датасета **Jenny** использовалось для трейна, во-вторых: для генерации аудио из датасета **LibriSpeech Test Clean**, возможно, использовался не сильно релевантный промт.
