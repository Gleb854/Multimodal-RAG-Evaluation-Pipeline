# PDF Q&A System (RAG over PDFs)

Локальный прототип системы вопросов–ответов по PDF: загрузка PDF → разбор на текст/картинки/таблицы → индексация в FAISS → поиск релевантных фрагментов → ответ LLM со ссылками на источники.

В проекте есть UI на Streamlit, CLI для ingest/query/migrate, а также модуль eval (датасет + метрики + LLM-as-a-Judge).

---

## Почему выбран такой подход

| Компонент | Причина выбора |
|-----------|----------------|
| **unstructured** | Практичный разбор PDF с учётом структуры (текст, таблицы, изображения); хорошо работает на разнородных научных/сканированных PDF |
| **FAISS** | Быстрый локальный векторный индекс без внешних сервисов. |
| **LangChain** | Удобные абстракции для документов/чанкинга/ретривинга и быстрые итерации |
| **Streamlit** | Быстрый путь к рабочему UI для загрузки файлов и диалога |
| **Eval + Judge smoke** | Бюджетная проверка, что judge реально отличает нормальный контекст от пустого/перемешанного |

---

## Структура проекта

```
pdf_qa_system/
├── src/
│   ├── ingestion/          # Ingest PDF (unstructured, чанкинг, картинки/таблицы)
│   │   ├── pdf_ingestor.py     # Главный класс PDFIngestor
│   │   ├── text_processor.py   # Сегментация текста
│   │   ├── table_processor.py  # Обработка таблиц (HTML→Markdown)
│   │   └── image_processor.py  # Извлечение картинок + Vision API captioning
│   ├── rag/                # RAG pipeline
│   │   ├── knowledge_base.py   # FAISS + OpenAI Embeddings
│   │   ├── retriever.py        # Smart routing (text/table/image)
│   │   └── qa_processor.py     # Генерация ответа с цитированием
│   ├── ui/
│   │   └── streamlit_app.py    # Web UI
│   ├── eval/               # Оценка качества
│   │   ├── dataset_generator.py  # Генерация датасета с gold_evidence
│   │   ├── runner.py             # Прогон eval
│   │   ├── judge.py              # LLM-as-a-Judge
│   │   ├── metrics.py            # Retrieval/answer метрики
│   │   └── report.py             # Генерация отчётов
│   └── config.py           # Настройки (pydantic-settings)
├── data/                   # Данные (в .gitignore)
│   ├── index/              # FAISS индексы
│   ├── uploads/            # Загруженные PDF
│   ├── processed/          # Обработанные чанки и картинки
│   └── eval/               # Eval-артефакты (датасеты, отчёты)
├── run.py                  # CLI (ingest, query, migrate-index, ui)
├── eval_cli.py             # Eval CLI (generate, run)
├── requirements.txt        # Python зависимости
└── environment.yml         # Conda environment
```

---

## Быстрый старт (Windows, без Docker)

### 1) Создать и активировать окружение

**Conda (рекомендуется):**

```bash
conda create -n pdf_qa python=3.11 -y
conda activate pdf_qa
pip install -r requirements.txt
```

**venv (альтернатива):**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Установить системные зависимости

Для корректной работы `unstructured[pdf]` нужны:

| Зависимость | Назначение | Установка (conda) |
|-------------|------------|-------------------|
| **Poppler** | PDF → изображения (pdftoppm) | `conda install -c conda-forge poppler` |
| **Tesseract OCR** | OCR для сканов | `conda install -c conda-forge tesseract` |

Проверка:

```bash
pdftoppm -h
tesseract -v
```

**Важно для Tesseract:** Если ошибка `TESSDATA_PREFIX`, установи переменную окружения:

```powershell
$env:TESSDATA_PREFIX = "C:\Users\<user>\miniconda3\envs\pdf_qa\share\tessdata"
```

### 3) Настроить переменные окружения

Создай `.env` в корне проекта:

```ini
OPENAI_API_KEY=sk-...
DATA_DIR=data

# Прокси (опционально)
USE_PROXY=false
PROXY_HOST=127.0.0.1
PROXY_PORT_HTTP=7890
PROXY_PORT_SOCKS5=7891
# PROXY_USER=
# PROXY_PASS=
```

`DATA_DIR` может быть абсолютным путём (например `D:\Projects\pdf_qa_system\data`).

### 4) Запуск

#### Web UI (Streamlit)

Из корня проекта:

**PowerShell:**

```powershell
$env:PYTHONPATH = (Get-Location).Path
streamlit run src/ui/streamlit_app.py
```

**CMD:**

```cmd
set PYTHONPATH=%cd%
streamlit run src/ui/streamlit_app.py
```

UI будет доступен по адресу: http://localhost:8501

#### CLI

```bash
# Запуск UI через CLI
python run.py ui --port 8501
```

---

## Использование

### Web UI (Streamlit)

1. Запусти UI (см. выше)
2. В сайдбаре укажи **OpenAI API Key**
3. Опционально включи **Use Proxy**
4. Два сценария:
   - **Load Index** — выбрать существующий индекс из `data/index/*`
   - **Upload PDFs → Build Index** — загрузить PDF и создать новый индекс
5. Задавай вопросы в чате

### CLI (`run.py`)

#### Ingest PDF → Build Index

```bash
python run.py ingest path\to\pdfs --index-name my_index
```

**Опции:**

| Флаг | Описание |
|------|----------|
| `--no-vision` | Отключить Vision API описания для картинок |
| `--proxy` | Использовать прокси для OpenAI |

Пример с прокси:

```bash
python run.py ingest D:\pdfs --index-name papers_index --proxy
```

#### Query по индексу

```bash
python run.py query "What is the main contribution?" --index-name my_index
```

#### Миграция старого индекса

Добавляет `chunk_id` в metadata (нужно для eval):

```bash
python run.py migrate-index --index test_v8_10 --out test_v8_10_migrated
```

---

## Eval (оценка качества)

Модуль eval позволяет:
1. **Сгенерировать датасет** — автоматически создать пары (вопрос, gold_evidence) из индекса
2. **Прогнать eval** — retrieval метрики + LLM-as-a-Judge

### 1) Генерация датасета

```bash
python eval_cli.py generate \
    --index-name test_v8_10_chunkid \
    --n 200 \
    --k-evidence 1 \
    --mix "text=0.6,image=0.25,table=0.15"
```

**Параметры:**

| Параметр | Описание | Default |
|----------|----------|---------|
| `--n` | Количество примеров | 200 |
| `--k-evidence` | Сколько evidence chunks на вопрос | 1 |
| `--mix` | Соотношение типов (text/image/table) | `text=0.6,image=0.25,table=0.15` |
| `--seed` | Random seed | 42 |
| `--max-per-source` | Лимит примеров на один PDF | 50 |
| `--resume` | Продолжить с checkpoint'а | — |
| `--proxy-enabled` | Использовать прокси | — |

Результат: `data/eval/<dataset_name>/dataset.jsonl`

### 2) Прогон eval

```bash
python eval_cli.py run \
    --index-name test_v8_10_chunkid \
    --dataset-path data/eval/<ds_name>/dataset.jsonl \
    --k 5
```

**Параметры:**

| Параметр | Описание | Default |
|----------|----------|---------|
| `--k` | Top-k retrieval (сколько чанков в контекст) | 5 |
| `--llm-model` | Модель для ответов | settings.llm_model |
| `--llm-temperature` | Temperature для ответов | 0.1 |
| `--judge-model` | Модель для Judge | llm-model |
| `--no-judge` | Отключить LLM-as-a-Judge | — |
| `--no-smart-routing` | Отключить smart_retrieve | — |
| `--use-source-scope` | Clamp retrieval к source_scope | — |
| `--strict-scope` | Не fallback если scope пуст | — |
| `--max-examples` | Лимит примеров из датасета | все |
| `--proxy-enabled` | Использовать прокси | — |

Результат: `data/eval/<run_name>/`
- `predictions.jsonl` — предсказания
- `report.md` — отчёт с метриками
- `report.json` — метрики в JSON

### Метрики

**Retrieval:**
- `recall@k` — доля примеров, где gold evidence найден в top-k
- `mrr` — Mean Reciprocal Rank (обратный ранг первого правильного результата)
- `ndcg@k` — Normalized Discounted Cumulative Gain

**Answer quality (LLM-as-a-Judge):**
- `faithfulness` — ответ основан только на контексте (нет галлюцинаций)
- `answer_relevance` — ответ релевантен вопросу
- `citation_support` — факты подкреплены ссылками на источники
- `refusal_correctness` — корректный отказ при отсутствии информации

### Как работает оценка

```
┌─────────────────────────────────────────────────────────────────┐
│                    ГЕНЕРАЦИЯ ДАТАСЕТА                           │
├─────────────────────────────────────────────────────────────────┤
│  1. Сэмплируем chunk из индекса (text/table/image)              │
│  2. LLM генерирует вопрос, ответ на который есть в этом chunk   │
│  3. Сохраняем: question + gold_evidence (chunk_id)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      ПРОГОН EVAL                                │
├─────────────────────────────────────────────────────────────────┤
│  Для каждого примера (question, gold_evidence):                 │
│                                                                 │
│  1. Retrieval: ищем top-k чанков по вопросу                     │
│     → recall: gold_evidence в top-k? (0/1)                      │
│     → mrr: на какой позиции gold_evidence?                      │
│                                                                 │
│  2. Generation: LLM генерирует ответ по найденному контексту    │
│                                                                 │
│  3. Judge: другой LLM оценивает качество ответа                 │
│     → faithfulness, relevance, citations (1-5 баллов)           │
└─────────────────────────────────────────────────────────────────┘
```

**Принцип LLM-as-a-Judge:**
- Judge получает: вопрос, контекст, сгенерированный ответ
- Оценивает по структурированным критериям (не знает "правильный" ответ)
- Выставляет баллы 1-5 по каждому критерию
- Smoke-тест проверяет, что judge отличает хороший контекст от пустого/перемешанного

### Результаты оценки (10 PDF, 30 примеров)

Тестовый прогон на 10 научных PDF (~600 чанков), k=5:

| Тип | N | Recall@5 | MRR@5 | nDCG@5 |
|-----|---|----------|-------|--------|
| **text** | 13 | 0.77 | 0.57 | 0.62 |
| **table** | 11 | 0.73 | 0.62 | 0.65 |
| **image** | 6 | 0.33 | 0.33 | 0.33 |
| **Overall** | 30 | **0.67** | **0.54** | **0.57** |

**Judge scores (LLM-as-a-Judge):**

| Метрика | Score |
|---------|-------|
| Faithfulness | 1.00 |
| Answer Relevance | 1.00 |
| Citation Support | 0.93 |
| Overall | 1.00 |

**Выводы:**
- Текст и таблицы извлекаются хорошо (recall ~0.75)
- Изображения — слабое место (recall 0.33), требуется улучшение vision descriptions
- Качество ответов высокое (faithfulness/relevance = 1.0)
- Citation support 0.93 — модель иногда не указывает источник явно

---

## Конфигурация

Все настройки через `.env` или переменные окружения:

| Переменная | Описание | Default |
|------------|----------|---------|
| `OPENAI_API_KEY` | API ключ OpenAI | — |
| `DATA_DIR` | Директория для данных | `data` |
| `LLM_MODEL` | Модель для ответов | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Модель эмбеддингов | `text-embedding-3-small` |
| `OPENAI_VISION_MODEL` | Модель для Vision API | `gpt-4o` |
| `CHUNK_SIZE` | Размер чанка (символы) | 1000 |
| `CHUNK_OVERLAP` | Перекрытие чанков | 200 |
| `RETRIEVAL_TOP_K` | Default top-k для retrieval | 5 |
| `USE_PROXY` | Включить прокси | `false` |
| `PROXY_HOST` | Хост прокси | — |
| `PROXY_PORT_HTTP` | HTTP порт | — |
| `PROXY_PORT_SOCKS5` | SOCKS5 порт | — |

---

## Архитектура

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   PDF(s)    │────▶│  PDFIngestor │────▶│  Documents  │
└─────────────┘     │  (unstructured)│    │  (chunks)   │
                    └──────────────┘     └──────┬──────┘
                                                │
                    ┌──────────────┐            │
                    │ KnowledgeBase│◀───────────┘
                    │   (FAISS)    │
                    └──────┬───────┘
                           │
┌─────────────┐     ┌──────▼───────┐     ┌─────────────┐
│   Query     │────▶│   Retriever  │────▶│  Context    │
└─────────────┘     │ (smart route)│     │  (top-k)    │
                    └──────────────┘     └──────┬──────┘
                                                │
                    ┌──────────────┐            │
                    │  QAProcessor │◀───────────┘
                    │   (LLM)      │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Answer +   │
                    │   Sources    │
                    └──────────────┘
```
