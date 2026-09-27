# Снимок результатов ветки `codex/day21-rotation-transfer`

Ветка не слита в `main` (нужен torch в рантайме). Для таблицы сравнения
вариантов модели (ТЗ 4) нужны её результаты, поэтому три файла скопированы
из коммита `f6ed090` как есть:

| файл здесь | путь в ветке |
|---|---|
| `local_aggregate_oof.csv` | `reports/rotation_local_transfer/aggregate_oof.csv` |
| `local_summary.json` | `reports/rotation_local_transfer/summary.json` |
| `full_summary.json` | `reports/rotation_transfer/summary.json` |

sha256:
    13e2a8c5315bda342b295b1d21db63d6b87633668d0bb6b1551ea91c3f3898ba  local_aggregate_oof.csv
    9d81095eaf8049e1d71ff90836263fea6792d33e70ed7108f43e481e1ef46aad  local_summary.json
    f4dddec73efcd182d61b61d3488d8063e5ad4cb0893c0ab8b8e2ffe763a781ff  full_summary.json
