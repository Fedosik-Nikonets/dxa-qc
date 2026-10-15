# tests/ — тесты

[← README](../README.md) · [Воспроизводимость](../docs/reproducibility.md)

```bash
python -m pytest tests -q        # окружение разработки (requirements.txt)
python run_tests.py              # то же без pytest — для машины без PyPI
```

Тесты не требуют данных организатора: всё, что может прийти в закрытом
наборе, собирается на синтетике генератором [`synth_dicom.py`](synth_dicom.py)
(написан на `struct`, без pydicom, — тест не зависит от библиотеки, которую
проверяет). Тесты, которым нужен `Data/`, при его отсутствии пропускаются.
Обе ветви ридера — с pydicom и встроенная — проверяются отдельно.

| группа | файлы |
|---|---|
| чтение и вход | `test_dicom_io.py` — загрузчик на синтетических DICOM всех профилей; `test_archive.py` — zip: кодировка имён, безопасная распаковка, уровень каталога исследований |
| области и маска | `test_routing.py` — роутер области и его края; `test_segmentation.py` — маска кости |
| правила | `test_spine.py`, `test_artifact.py`, `test_femur.py`, `test_medial_track.py`, `test_rotation_profile.py`, `test_rotation_rule.py`, `test_trochanter.py`, `test_rotation.py` |
| балл и метрики | `test_score.py` — `quality_score` согласован с решением; `test_metrics.py` — метрики ТЗ 8.4 против ручного расчёта |
| выход | `test_report.py` — схема ТЗ 2.5 и побайтовая запись; `test_overlay.py`, `test_dicom_sc.py`, `test_dicom_sr.py` — overlay-серия, DICOM SC и SR |
| сквозной прогон и сервис | `test_batch.py` — состав таблицы, `Failure`, дедупликация, детерминизм; `test_service.py` — HTTP API; `test_webui.py` — веб-интерфейс |
| устойчивость | `test_stress.py` — стресс-корпус: ни одного вылета, перепаковки не меняют решений |
| честность проверки | `test_split.py` — разбиение; `test_checkpoint_metrics.py`, `test_multilabel_head.py` — порог и голова не видят меток отложенного фолда; `test_femur_landmarks.py` — целостность разметки точек |

На заморозке `v1.1.0` в публичной копии без `Data/`: 301 тест пройден,
0 упало, 12 пропущено.
