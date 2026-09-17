# EXPLANATION.md — как устроен код `bsdesolve`

> Документ описывает **текущую реализацию** (v0.1.0, Фазы 0–1: Core BSDE — Euler-Maruyama).
> Все идентификаторы кода оставлены на английском (как в исходниках), текст — на русском.

---

## 1. Что решает библиотека

`bsdesolve` решает **BSDE** (Backward SDE) и **FBSDE** (Forward-Backward SDE):

```
BSDE:   dY_t = -f(t, Y_t, Z_t) dt + Z_t dW_t,      Y_T = ξ
FBSDE:  dX_t =  μ(t, X_t) dt + σ(t, X_t) dW_t      (forward, заданное x0)
        dY_t = -f(t, X_t, Y_t, Z_t) dt + Z_t dW_t  (backward, заданное ξ)
```

На выходе: **Y** (стоимостный процесс, «ценовой» компонент) и **Z** (диффузионный/хеджующий процесс).
Практический смысл: `Y_0` — это цена (напр., опциона), `Z` — delta-хедж.

Сейчас реализована **одна схема** — `Euler` (Monte Carlo + OLS-регрессия, «backward Euler-Maruyama»).
`Deep`, `Milstein`, FBSDE-strongly-coupled и проч. — запланированы (см. PLAN.md).

---

## 2. Структура пакета

```
bsdesolve/
├── __init__.py            # публичный API: solve, solve_fbsde, все классы, __version__
├── core/                  # «доменные» определения (без численного решения)
│   ├── problem.py         # BSDEProblem, FBSDEProblem  (dataclass-контейнеры задачи)
│   ├── generator.py       # Generator (ABC) + LinearGenerator, NonlinearGenerator
│   ├── terminal.py        # TerminalCondition (ABC) + Constant/Function/PathDependent
│   ├── sde.py             # ForwardSDE (ABC) + GBM, OU, CIR, BlackScholes
│   └── solver.py          # solve(), solve_fbsde() — точки входа + реестр схем
├── schemes/               # численные схемы (алгоритмы решения)
│   ├── scheme.py          # Scheme (ABC) + BSDEResult (dataclass-результат)
│   └── euler.py           # EulerScheme — единственная реализованная схема
├── diagnostics/
│   └── convergence.py     # ConvergenceReport (метрики сходимости, summary)
├── benchmarks/
│   └── __init__.py        # blackscholes(), linear_bsde(), barenblatt() + Benchmark
├── deep/                  # (заглушка, Фаза 2)
└── utils/                 # (заглушка)
```

**Принцип разделения:** `core/` описывает *что* решать (генератор, терминал, SDE, проблема),
`schemes/` — *как* решать (алгоритм). Точки входа (`solver.py`) связывают их через реестр.

---

## 3. Ядро: абстракции (core/)

### 3.1 `Generator` (`core/generator.py`) — генератор `f(t, y, z)`

Класс-база для «дрейфа» BSDE. Контракт — один метод:

```python
Generator.__call__(t, y, z) -> Tensor   # вычисляет f(t, y, z), форма (N,)
```

| Класс | Что делает |
|---|---|
| `LinearGenerator(a, b, c)` | `f = a(t)·y + b(t)·z + c(t)`. Коэффициенты `a,b,c` могут быть `float`, `callable(t)` или `Tensor`. Самый частый случай — `LinearGenerator(a=r)` (безрисковая ставка). |
| `NonlinearGenerator(func)` | Обёртка над пользовательской функцией `func(t, y, z) -> Tensor`. |

**Важный нюанс по shape:** `LinearGenerator.__call__` всегда возвращает `(N,)` (один скаляр на путь).
Если `z` имеет форму `(N, dim_w)`, он **редуцируется по последней оси** (`z.sum(dim=-1)`) до `(N,)`.
Это согласовывает shapes: `a*y` → `(N,)`, `b*z_reduced` → `(N,)` → сумма → `(N,)`.

> Тип-алиас `Coeff = float | Callable[[Tensor], Tensor] | Tensor` — аннотация для коэффициентов
> (ruff требует `X | Y`, не `Union`).

### 3.2 `TerminalCondition` (`core/terminal.py`) — терминал `ξ`

Контракт: `__call__(x, t) -> Tensor`, где `x` — терминальное состояние (форма `(N,)` или `(N, dim)`).

| Класс | Что делает |
|---|---|
| `ConstantTerminal(value)` | `ξ = const` (не зависит от `x`). |
| `FunctionTerminal(func)` | `ξ = g(x_T)` — функция от терминального состояния (напр. call: `max(x-K,0)`). |
| `PathDependentTerminal(func)` | Платёж, зависящий от всей траектории. **Сейчас заглушка**: фактически вызывает `func(x)` (только финальный шаг), полная поддержка — позже. |

### 3.3 `ForwardSDE` (`core/sde.py`) — forward-процесс

Контракт: `drift(t, x) -> (N, dim)`, `diffusion(t, x) -> (N, dim, dim_w)`, плюс **готовый**
`simulate(t_grid, num_paths, ...) -> (N, T+1, dim)` (Euler-Maruyama, вперёд).

| Класс | Процесс |
|---|---|
| `GBM(drift, diffusion, x0)` | `dS = μS dt + σS dW` (акции). |
| `OU(theta, mu, sigma, x0)` | `dX = θ(μ−X) dt + σ dW` (mean-reversion, ставки/волатильность). |
| `CIR(theta, mu, sigma, x0)` | `dX = θ(μ−X) dt + σ√X dW` (square-root, variance). |
| `BlackScholes(r, sigma, x0)` | `GBM` с риск-нейтральным дрейфом `r` (удобный сахар для прайсинга). |

> Примечание: в `EulerScheme` forward-симуляция делается **внутри схемы** (своим циклом), а не через
> `ForwardSDE.simulate` — чтобы схема полностью контролировала приращения `ΔW` и использовала их же для
> регрессии `Z`. Метод `simulate()` нужен, если хочется посчитать траектории отдельно.

### 3.4 `BSDEProblem` / `FBSDEProblem` (`core/problem.py`)

Простые `@dataclass`-контейнеры, собирающие всё вместе:

```python
BSDEProblem(generator, terminal_condition, sde=None, t_span=(0,1), dim=1, name="bsde")
FBSDEProblem(generator, terminal_condition, sde, t_span=(0,1), dim=1, coupling="weak", name="fbsde")
```

Сами по себе ничего не решают — это «пакет параметров», который передаётся в схему.
`FBSDEProblem` сейчас используется **только как тип**; `solve_fbsde()` фактически строит `BSDEProblem`.

---

## 4. Точки входа (`core/solver.py`)

### `solve(...)` — главный API

```python
solve(generator, terminal_condition, sde=None, t_span=(0.0,1.0), dim=1,
      method="euler", num_paths=10_000, num_time_steps=100,
      device="cpu", seed=None, return_diagnostics=False) -> BSDEResult
```

**Что делает:**
1. Валидирует `method` по реестру `_SCHEMES`.
2. Оборачивает аргументы в `BSDEProblem`.
3. Создаёт схему: `scheme = _SCHEMES[method]()` (например `EulerScheme()`).
4. Вызывает `scheme.solve(problem, num_paths, ...)` и возвращает `BSDEResult`.

### `solve_fbsde(...)`

Аналогичный API, но требует `sde` и принимает `coupling="weak"|"strong"`.
**Сейчас (MVP):** weakly-coupled FBSDE = обычный BSDE с известным `X` (генератор сам работает с `X`),
т.е. сводится к тому же `EulerScheme`. Strongly-coupled (Picard-итерация) — ещё не реализован.

### Реестр схем

```python
_SCHEMES: dict[str, type[Scheme]] = {"euler": EulerScheme}
```

Чтобы добавить новую схему — вписать её сюда (и реализовать `Scheme`). См. §7.

---

## 5. Результат: `BSDEResult` (`schemes/scheme.py`)

```python
@dataclass
class BSDEResult:
    Y:  Tensor   # (num_paths, T+1)   — стоимостный процесс; Y[:,i] при t_grid[i]
    Z:  Tensor   # (num_paths, T, dim_w) — хедж; Z[:,i] на интервале [t_i, t_{i+1}]
    X:  Tensor | None  # (num_paths, T+1, dim) — forward-траектории (если есть SDE), иначе None
    convergence: ConvergenceReport | None
    t_grid: Tensor | None   # (T+1,)

    @property Y0       -> Tensor  # Y[:,0], (num_paths,)
    @property Y0_mean  -> float   # MC-оценка цены = Y[:,0].mean()
```

`Y0_mean` — это и есть «ответ» (цена в момент t=0). `Z` — хедж-стратегия (delta).

---

## 6. Схема `EulerScheme` (`schemes/euler.py`) — сердце библиотеки

Алгоритм **backward Euler-Maruyama + OLS-регрессия** (Bally–Crisan–Delarue; линейный случай — Han–Jentzen–E).

### Параметры
- `dt = (T - t0) / num_time_steps`, сетка `t_grid = linspace(t0, T, T+1)`.
- `dim_w` (размерность броуновского движения) = размерность SDE (стандартный Markov-случай), иначе 1.

### Шаги

**Шаг 1. Симуляция forward + запись приращений `ΔW`.**
```python
dW = torch.randn(num_paths, T_steps, dim_w) * sqrt(dt)   # dW[:,i] — приращение за [t_i,t_{i+1}]
```
Если есть SDE — идём вперёд Euler-Maruyama: `X_{i+1} = X_i + μ dt + σ·ΔW_i`
(`torch.einsum("nab,nb->na", sigma, dW)` — умножение матрицы волатильности на приращение).
Если SDE нет — чистая броуновская (W = cumsum(ΔW)).

**Шаг 2. Терминальное условие.**
```python
X_T = X_paths[:, -1]
Y_T = terminal_condition(X_T, t_grid[-1])   # (N,)
Y[:, -1] = Y_T
```

**Шаг 3. Обратная итерация** — `for i in range(T_steps - 1, -1, -1)` (от T-1 к 0):
```python
Y_next = Y[:, i+1]                          # (N,)
Z_next = Z[:, i+1]  (или 0 в последнем шаге) # (N, dim_w)
f_val  = generator(t_i, Y_next, Z_next)      # (N,)
target = Y_next - f_val * dt                 # (N,)  — «куда надо дойти» за dt

Z_i, Y_i = _ols_regression(target, dW[:, i]) # регрессия target на ΔW_i
Y[:, i] = Y_i;  Z[:, i] = Z_i
```

### `_ols_regression(target, dW) -> (Z, Y)`

Классическая OLS в **центрированной** форме. Моделируем `target ≈ Y + Zᵀ·ΔW`, где `Z` — вектор
(размер `dim_w`), `Y` — скаляр:
```python
A = (dW_cᵀ dW_c)/N          # (dim_w, dim_w)
b = (dW_cᵀ target_c)/N      # (dim_w,)
Z = solve(A + 1e-8·I, b)    # регуляризация; fallback — pinv
Y = mean(target) - Zᵀ·mean(ΔW)
```

### ⚠️ Критическое свойство: **глобальная (не по-траекторная) регрессия**

Это **самое важное** для понимания. На каждом шаге делается **одна** OLS-регрессия по **всем** N путям
одновременно → получаются **один** скаляр `Y_i` и **один** вектор `Z_i`, которые **копируются во все пути**.
Проверено эмпирически:

```
t_grid:      [0.0,   0.2,   0.4,   0.6,   0.8,   1.0]
Y (по шагу): 10.37, 10.48, 10.58, 10.69, 10.80, 11.02   <- std=0 (все пути одинаковы)
Z (по шагу):  ~0,     ~0,     ~0,     ~0,    10.85      <- std=0 (все пути одинаковы)
```

- `Y[:, i]` одинаков по всем путям (std = 0) для i < T; **только** `Y_T` (финальный) зависит от пути.
- `Z[:, i]` тоже одинаков по всем путям.
- Это значит, что текущая реализация **не** аппроксимирует марковское решение `Y_t = V(t, X_t)` по каждой
  траектории, а даёт **усреднённые** (глобальные) значения. Для задач, где ответ детерминирован (цена,
  `Y_0`) это **работает и сходится** (BS call: 10.37–10.49 vs exact 10.45). Но `Z` как локальный
  delta-хедж `∂V/∂x` — аппроксимируется слабо (≈0 вдали от T).

> **Почему так:** это упрощение «Monte Carlo + регрессия». Чтобы получать `Z(t,x)` как функцию состояния,
> нужно либо регрессировать по подвыборке (partitioning/regression on state), либо использовать NN
> (Deep BSDE, Фаза 2). Сейчас — базовый вариант, честный для `Y_0`, но не для точного `Z(t,x)`.
> Это **известное ограничение**, задокументированное в NOTES.md.

### Сходимость
- По времени: O(dt). По числу путей: O(1/√N) (MC-ошибка).
- Для `Y_0` (цены) — MC-шум мал при больших N (т.к. `Y_0` почти детерминирован для Markov SDE с
  фиксированным `x0`).

---

## 7. Диагностика: `ConvergenceReport` (`diagnostics/convergence.py`)

`@dataclass`, строится, если `return_diagnostics=True`:

| Поле | Смысл |
|---|---|
| `mc_std_err` | MC-ошибка `Y_0` = std(Y_0)/√N |
| `y0_mean`, `y0_std` | среднее/дисперсия `Y_0` |
| `step_errors` | std «target» на каждом шаге (прокси дискретизационной ошибки) |
| `y_means`, `y_vars` | средние/дисперсии Y по шагам |
| `converged` | эвристика: `mc_std_err < 1% · |y0_mean|` |
| `summary()` | человек-читаемая строка-отчёт |

> `converged` — **грубая эвристика**. Для строгой проверки нужно сравнивать с известным exact-решением
> (что и делают тесты/бенчмарки).

---

## 8. Бенчмарки (`benchmarks/__init__.py`)

Функции-фабрики возвращают `Benchmark` (dataclass: `name`, `description`, `make_problem()`, `exact(t,x)`,
`params`). `exact` — замкнутая формула для валидации.

| Бенчмарк | BSDE | Exact |
|---|---|---|
| `blackscholes(S0,K,r,sigma,T)` | `dY=-rY dt+Z dW`, `Y_T=max(S_T-K,0)`, SDE=GBM | Формула Black-Scholes call (через `erf`). |
| `linear_bsde(r,T,xi)` | `dY=-rY dt+Z dW`, `Y_T=xi` (const, нет SDE) | `Y_t = xi·exp(-r(T-t))`, `Z=0`. Простейший регрессионный тест. |
| `barenblatt(d,T)` | нелинейный генератор по Z | **Заглушка** (нет closed-form; полная реализация — Фаза 2). `exact` временно возвращает 1.0. |

`get_all_benchmarks()` — список всех.

---

## 9. Конкретный прогон (Black-Scholes call)

`examples/example_black_scholes.py`:

```python
sde      = GBM(drift=r, diffusion=sigma, x0=S0)          # dS = rS dt + σS dW
generator = LinearGenerator(a=r)                          # f(t,y,z) = r·y
terminal = FunctionTerminal(lambda x: torch.clamp(x-K, min=0))   # ξ = max(S_T-K,0)

result = solve(generator, terminal, sde=sde,
               t_span=(0.0, T), num_paths=100_000, num_time_steps=200, seed=42,
               return_diagnostics=True)
print(result.Y0_mean)   # ≈ 10.45 (exact 10.4506)
```

Цепочка вызовов:
```
solve()  →  BSDEProblem(...)  →  _SCHEMES["euler"]() = EulerScheme()
         →  scheme.solve(problem, N, T, ...)
              1) симуляция X + ΔW
              2) Y_T = ξ(X_T)
              3) обратная итерация + OLS на каждом шаге
         →  BSDEResult(Y, Z, X, t_grid, convergence)
```

---

## 10. Как расширять (добавить новую схему)

1. В `schemes/` создайте `myscheme.py` с классом `MyScheme(Scheme)`, реализуйте
   `solve(problem, num_paths, num_time_steps, device, seed, return_diagnostics) -> BSDEResult`.
   `problem` содержит всё: `generator`, `terminal_condition`, `sde`, `t_span`, `dim`.
2. В `schemes/__init__.py` экспортируйте его.
3. В `core/solver.py` добавьте в реестр: `_SCHEMES = {"euler": EulerScheme, "deep": MyScheme}`.
4. Публичный `solve(..., method="deep")` заработает автоматически.

Для **Deep BSDE** (Фаза 2) — NN аппроксимирует `Z(t,x)` (марковское), ретроградная итерация,
loss = терминальный + MSE по шагам. Это решит ограничение глобальной регрессии из §6.

---

## 11. Известные ограничения (текущая v0.1.0)

1. **Глобальная регрессия** (§6): `Y_i`, `Z_i` одинаковы по всем путям; `Z(t,x)` как функция состояния
   не аппроксимируется. Цена (`Y_0`) сходится, но точный локальный хедж — нет. → Deep BSDE (Фаза 2).
2. **Одна схема** (`euler`). Нет `milstein`, `deep`, `regression`.
3. **FBSDE only weakly-coupled.** Strongly-coupled (Picard) — не реализован.
4. **Скалярный Y** (одна компонента). Vector BSDE — Фаза 5.
5. **PathDependentTerminal — заглушка** (только финальный шаг).
6. **Без прыжков/Lévy** (Фаза 4), **без G-BSDE** (Фаза 6).
7. `ForwardSDE.simulate()` дублирует логику из `EulerScheme` (симуляция в схеме — своя).

---

## 12. Тесты (`tests/test_euler.py`) — 9 passed

- `TestLinearBSDE` — точное решение `xi·exp(-r(T-t))`, `Z≈0`.
- `TestBlackScholes` — call price, put-call parity.
- `TestDiagnostics` — отчёт возвращается.
- `TestSDE` — среднее GBM / OU (mean reversion).
- `TestEdgeCases` — неизвестный `method` → ошибка; `num_time_steps=1`.

Запуск: `make test` (= `uv run pytest tests/ -v`).


