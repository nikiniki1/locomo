# locomo_gen

Пайплайн генерации синтетических бенчмарков на основе [LoCoMo](https://github.com/nikiniki1/locomo).
Генерирует сценарии долгосрочных диалогов с QA-парами для оценки систем памяти.

## Что изменено относительно оригинала

- **Поддержка GigaChat** (`--no-structured-output`): переключает генерацию на текстовый fallback для моделей без поддержки OpenAI structured output
- **Батчевая генерация QA** (`generation/stages/qa.py`): генерирует вопросы по сессиям поочерёдно вместо одного большого вызова — предотвращает обрезание ответа у моделей с ограниченным output
- **Защита от зависания** (`generation/services/event_service.py`): прерывает бесконечный цикл, если модель возвращает непарсируемый ответ; санитизация ID для нестандартных форматов
- **Безопасный парсинг** (`generation/stages/qa.py`): возвращает пустой результат вместо исключения при ошибке парсинга
- **`locomo_to_spade.py`**: конвертирует сгенерированный benchmark JSON в YAML-формат для [spade-llm bench](https://github.com/AKBAPEL/spade-llm)

## Установка

```bash
cp .env.example .env  # заполнить API-ключ и base URL
pip install -r requirements.txt
```

## Генерация сценариев

```bash
python generate_benchmark.py \
  --out-dir outputs/my_run \
  --language ru \
  --num-samples 5 \
  --num-sessions 20 \
  --num-events 15 \
  --qa-per-sample 24 \
  --model openai/gpt-4o

# Для GigaChat и других моделей без structured output:
python generate_benchmark.py ... --no-structured-output
```

## Конвертация в формат spade-llm

```bash
python locomo_to_spade.py outputs/my_run/benchmark.json --out-dir ../locomo_scenarios/
```

Папка `outputs/` добавлена в `.gitignore`. Необходимые переменные окружения — в `env.example`.
