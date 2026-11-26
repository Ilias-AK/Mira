# Кандидат v2: интеграция пакета mira_delivery_clean_v2.1

Ветка `train/delivery-v2`. Рабочая Мира (`config/`, `runs/mira_intent/intent.pt`) не изменена.

## Что сделано

| Шаг | Файл |
|---|---|
| Проверка канона в репликах (регулярные выражения, без моделей) | `mira/canon_check.py` |
| Воспроизводимый импорт ZIP → формат проекта + manifest SHA-256 | `mira/import_delivery.py` |
| Кандидатные профили runtime | `config/candidate_v2/` (данные пакета как есть), `config/candidate_v2_keep/` (правила и новые темы пакета, но утверждённые тексты ответов из рабочего профиля) |
| Данные для обучения и проверки | `data/delivery_v2/` (см. `manifest.json`) |
| Обучение нейросети тем с внешней проверкой | `mira/train_intent.py --extra-train ... --val-file ...` |
| Оценка Миры целиком, нейросети тем и генератора | `mira/eval_candidate.py`, `mira/eval_intent_model.py`, `mira/eval_generator.py` |
| Наборы регрессии и памяти | `mira/eval_sets/regression.json`, `mira/eval_sets/memory.json` |
| Полный прогон на ноутбуке | `mira/train_v2.ps1` |

## Как используются файлы архива

| Файл архива | Использование |
|---|---|
| runtime/*.clean.json | профиль `config/candidate_v2*` |
| train/paraphrases_train.jsonl | обучение нейросети тем |
| train/context_followups.jsonl | обучение нейросети тем + формулировки уточнений в team_qa кандидата |
| train/new_topics_retrieval_clean.jsonl | только поиск по примерам (`source=retrieval_v2`, порог похожести 0.75), не классы |
| train/new_topics_intent_candidates.jsonl | не используется: `data/delivery_v2/intent_candidates_review.jsonl` на ревью |
| train/dialogues_*_train.jsonl, traps_train.jsonl | генератор: одна реплика Миры = пример, контекст ≤ 8, общий group_id |
| validation/* | только оценка |
| excluded/, reports/, generator_ready/, canon_extensions.json | не используются (импортёр запрещает их чтение) |

Для выбора чекпойнта генератора используется `dev_from_train.jsonl` (5 % групп train), поэтому validation архива остаётся независимой.

## Запуск

```powershell
# весь прогон (нейросеть тем ~15 мин + генератор до 1,5 ч), рабочую Миру не трогает
powershell -ExecutionPolicy Bypass -File D:\turing-arena\mira\train_v2.ps1
# без генератора
powershell -ExecutionPolicy Bypass -File D:\turing-arena\mira\train_v2.ps1 -SkipGenerator
# пересобрать данные из архива (архив только читается)
python -m mira.import_delivery --zip D:\path\mira_delivery_clean_v2.1.zip
# поговорить с кандидатом
python -m mira.chat --config-dir config\candidate_v2_keep --intent runs\mira_intent_v2\intent.pt
```

Выпуск в рабочую версию — только вручную и только если `runs/eval_v2/comparison_*.md` показывает, что кандидат не хуже по критическим проверкам и лучше по качеству.

## Итог (26.11.2026)

Подтверждающий прогон на ноутбуке (`runs/train_v2/eval-20261126-0144`): кандидат `candidate_v2_keep` + нейросеть тем v2
лучше baseline по 10 метрикам качества, хуже ни по одной; регрессия 29/29, память 14/14, канон 0, не-русский 0,
> 300 символов 0, > 45 с 0. Нейросеть тем отдельно: точность 0.695 → 0.838, macro-F1 0.672 → 0.822 (validation архива, synthetic).

Выпуск (ветка `release/v2`): `config/` = `candidate_v2_keep` + метка `_profile`, `runs/mira_intent/intent.pt` = v2.
После проверки добавлено поле `pin` (hall_study повторяется как team_uni) — проверено без нейросети: самопроверка 12/12,
регрессия 29/29, память 14/14.

- откат: `mira/rollback_v1.ps1` (копия v1 в `runs/baseline_v1`);
- `build/make_bank.py` и `mira/team_qa.py` больше не затирают выпущенный профиль (пишут `*.built.json`, перезапись — `--force`);
- `mira/train_intent.ps1` учит по рецепту v2 в `runs/mira_intent_next` и только сравнивает;
- генератор v2 (`runs/generator_v2` на ноутбуке) к ответам Миры не подключён.
