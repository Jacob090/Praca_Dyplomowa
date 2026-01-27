# Autonomous 4-DOF Robotic Arm RL (MuJoCo + SAC)

Kompletny, działający projekt do pracy dyplomowej: autonomiczne sterowanie 4‑osiowym ramieniem (yaw + 3x pitch) z chwytakiem bez kontroli orientacji. Zadanie: pick‑and‑place w MuJoCo z curriculum learning (3 etapy).

## Założenia i decyzje projektowe

- **Model fizyczny oraz przegubowy wzorowany na publicznych modelach:** Universal Robots UR5e (Ramie robotyczne 6DOF), Robotiq 2F-85 (Chytak). Makieta sceny w pełni autorska.
- **Manipulator 4 DOF**: 4 przeguby zawiasowe w osi Z (yaw) i osi Y (pitch). Chwytak to osobny "staw" (nie liczony do DOF), sterowany jedną zmienną akcji.
- **Brak orientacji TCP**: obserwacje i nagrody nie używają kwaternionów ani orientacji.
- **Chwytanie**: uproszczone „soft‑grasp”. Gdy chwytak zamknięty i TCP blisko obiektu, obiekt jest „przyklejany” do TCP poprzez bezpośrednie ustawianie pozycji w `MjData` (stabilne, deterministyczne i wystarczające do celów RL bez komplikacji kontaktów).
- **Curriculum learning**:
  - Stage 1 (reach): tylko zbliżenie TCP do obiektu -> korekta.
  - Stage 2 (grasp): wczytanie wag z Stage 1 i bonusy za zamknięcie chwytaka i podniesienie.
  - Stage 3 (place): wczytanie wag z Stage 2 i nagrody za przeniesienie do strefy celu.
- **Deterministyczne seedy**: `reset(seed)` używa `gymnasium.utils.seeding`.
- **Stabilność**: akcje są clipowane, a prędkości stawów ograniczane ze względu na zaimplementowaną fizykę ramienia.


## Instalacja

```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uruchomienie środowiska i losowe akcje

```
python test_env.py --stage reach --steps 200 --render
```

## Trening SAC z curriculum learning **!!!uruchamiać tylko raz do wygenerowania scieżek!!!**

### Stage 1 – Reach
```
python training/train_stage1_reach.py
```

### Stage 2 – Grasp (ładuje wagi z Stage 1)
```
python training/train_stage2_grasp.py
```

### Stage 3 – Place (ładuje wagi z Stage 2)
```
python training/train_stage3_place.py
```

## Wznawianie treningu z checkpointów

```
python training/train_stage1_reach.py --resume evals/stage1/best_model.zip
python training/train_stage2_grasp.py --resume evals/stage1/best_model.zip
python training/train_stage3_place.py --resume evals/stage1/best_model.zip
```

## TensorBoard

```
tensorboard --logdir runs
```

## Ewaluacja polityki

```
python evaluation/evaluate_policy.py --model-path evals/stage1/best_model.zip --stage place --episodes 10 --render
```

## Opis obserwacji

Wektor obserwacji:

1. Pozycje 4 przegubów (qpos)
2. Prędkości 4 przegubów (qvel)
3. TCP xyz
4. Pozycja obiektu xyz
5. Prędkość liniowa obiektu xyz
6. Dystans TCP → obiekt
7. Dystans obiekt → cel

## Termination / Truncation

- **terminated**:
  - sukces zadania (zależnie od etapu)
  - NaN w obserwacji
  - obiekt poza workspace
- **truncated**:
  - limit kroków

## Renderowanie (viewer jest tylko do podlądu fizycznego)

- `render_mode="human"`: interaktywny podgląd MuJoCo.
- `render_mode="rgb_array"`: zwraca klatkę RGB.


## Konfiguracja

Parametry środowiska i treningu są w `configs/`. Najważniejsze:

- `configs/env.yaml` – sceny, spawn obiektu i celu, progi nagród.
- `configs/sac_stage*.yaml` – hiperparametry SAC, częstotliwości checkpointów i ewaluacji.

## Uwagi końcowe

- Chwytak nie ma orientacji – TCP zawsze pionowo w dół (ostatnia oś do korekcji).
- Brak quaternionów w obserwacjach i nagrodach.
- Projekt zgodny z wymaganiami: MuJoCo 3.4.0, gymnasium 1.2.3, stable-baselines3 2.7.1.

## TODO

- Dopracować tuning nagród i progów dla stabilniejszej zbieżności.
- Przeprowadzić pełne treningi dla wszystkich etapów i zapisać wyniki (zapis automatyczny do bazy w trakcie pracy).
- Przygotować zestaw eksperymentów porównawczych (hiperparametry, progi, randomizacja - 50% spełnione).
- Rozważyć bardziej realistyczny chwyt (kontakty) zamiast soft‑grasp. (nie obowiązkowe ale moze coś poprawi w precyzji - do sprawdzenia)
- Dodać dodatkowe metryki do ewaluacji (np. czas sukcesu, średnia odległość).
- Zmienić miejsca do treningu: symulacja MuJoCo -> CPU, Uczenie sieci SAC -> GPU (render_mode=None)

## Terminologia

-TCP - punkt styczności, roboczy narzędzia robota
