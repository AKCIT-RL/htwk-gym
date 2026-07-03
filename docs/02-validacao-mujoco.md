# Validação cross-sim no MuJoCo (roda no Mac ✅)

## Por quê
Mesma prática do paper e do htwk-gym: treinar no Isaac Gym (GPU remota) e validar a
política no MuJoCo antes do robô real. MuJoCo tem solver de contatos diferente/mais fiel —
se a política sobrevive à troca de simulador, a chance de funcionar no T1 real sobe muito.
É também o demo que a Tsinghua mostrava no notebook.

## Viabilidade no Mac (verificado em 2026-07-03)
- MuJoCo 3.10 instala nativo em Apple Silicon (`pip install mujoco`), com viewer.
- T1_locomotion.xml do htwk-gym carrega: 19 qpos, 12 atuadores.
- Benchmark no MacBook Neo (A18 Pro): 5000 steps em 0.07 s ≈ 140× tempo real p/ 1 env.
  Sobra para viewer interativo e avaliação serial (grid de gols leva minutos).

## O que já existe para reaproveitar (htwk-gym)
- `resources/T1/T1_locomotion.xml` e `T1_serial.xml` (MJCF do robô)
- `play_mujoco_kick.py`: carrega policy JIT/pth, injeta BOLA no MJCF em runtime,
  viewer interativo + modo `--evaluate` (mesmo layout de artefatos do Isaac)
- helpers numpy de quaternion/frames

## O ambiente Soccer no MuJoCo — componentes

### 1. Cena `soccer_field.xml` (MJCF)
```xml
<mujoco>
  <include file="T1_locomotion.xml"/>       <!-- robô -->
  <worldbody>
    <geom name="field" type="plane" size="8 5.5 0.1" friction="0.8 0.005 0.0001"/>
    <!-- linhas do campo: sites/texturas (visual apenas) -->
    <body name="ball" pos="1 0 0.11">
      <freejoint/>
      <geom type="sphere" size="0.11" mass="0.43"
            friction="0.6 0.008 0.0001" solref="0.02 0.6"/>  <!-- Size 5 -->
    </body>
    <!-- gols: 2 traves + travessão como capsules em x=±7, |y|<1.3 (com colisão!) -->
  </worldbody>
</mujoco>
```
- Campo 14×9 m (plane 8×5.5 dá margem de out).
- Gol = geoms capsule (no MuJoCo vale ter trave física; no Isaac era só região).
- Calibrar `solref/solimp` e fricções da bola p/ quicar como bola real (teste de queda:
  altura do quique ≈ 60-65% p/ Size 5 em grama curta).

### 2. Wrapper `mujoco_soccer_env.py`
Replica EXATAMENTE a interface de observação do env Isaac (49 dims):
- mesmos índices/ordem/normalização de obs (fonte única: gerar constantes de um módulo
  compartilhado `soccer_obs.py` usado pelos DOIS envs — evita drift Isaac↔MuJoCo)
- PD control igual (kp/kd por junta, decimation 10, dt=0.002 → 50 Hz de política)
- ball obs pelo MESMO caminho `_get_ball_observation()` (ruído/virtual perception)
- goal pos/dir no frame do robô

### 3. `play_mujoco_soccer.py` (viewer)
Baseado no play_mujoco_kick.py:
- carrega checkpoint exportado (JIT `.pt`)
- teclas: R reset tudo, B re-posiciona bola, G move gol-alvo, setas empurram a bola
  (simular "adversário"), P imprime estado
- overlay: estimativa da bola pela política (quando tivermos o decoder, F3)

### 4. `evaluate_mujoco_soccer.py` (headless)
Mesmo grid da Fig. 3A do paper: bola em células fixas do campo, robô no centro,
N tentativas/célula → taxa de gol, quedas, tempo até contato → heatmap.
Comparar lado a lado com o mesmo harness no Isaac (F1.7): a DIFERENÇA entre os dois
heatmaps é a métrica de gap sim-to-sim.

## Passo a passo de implementação

| # | Tarefa | Validação |
|---|---|---|
| MJ.1 | venv Python no Mac + `pip install mujoco torch numpy pyyaml` | `import mujoco` ok (✅ já testado) |
| MJ.2 | `soccer_field.xml`: campo + bola Size 5 + gols | viewer: bola quica ~60-65%, rola e para; robô em pé sobre o plano |
| MJ.3 | módulo compartilhado `soccer_obs.py` (layout de obs usado por Isaac E MuJoCo) | teste unitário: shapes/índices idênticos nos dois builders |
| MJ.4 | wrapper env + PD 50 Hz | política DUMMY (zeros) → robô fica em pé parado; smoke test 1000 steps sem NaN |
| MJ.5 | `play_mujoco_soccer.py` viewer + teclas | dirigir uma política de walk existente do htwk-gym no campo (sanity) |
| MJ.6 | `evaluate_mujoco_soccer.py` headless + heatmap | roda o grid completo em < 30 min no Mac |
| MJ.7 | ponte de checkpoint: export JIT do treino Isaac → play no Mac | política F1.6 (baseline) caminha até a bola no MuJoCo |

Dependências: MJ.1–MJ.3 podem começar JÁ (antes mesmo do env Isaac ficar pronto —
aliás o MJ.3 é insumo da F1.3). MJ.7 depende do primeiro checkpoint da F1.6.

## Armadilhas conhecidas (do próprio play_mujoco_kick.py)
- Ordem das juntas Isaac ≠ MuJoCo: mapear por NOME, nunca por índice.
- Quaternion: Isaac usa xyzw, MuJoCo usa wxyz — converter sempre.
- Default dof pos e action_scale precisam bater com o yaml do treino.
- decimation/dt diferentes entre sims quebram silenciosamente a política — fixar 500 Hz
  física / 50 Hz política nos dois.

---

## Status de implementação (2026-07-03)

### ✅ MJ.1 — Ambiente Python (Mac)
`python3 -m venv .venv && pip install -r requirements.txt`. MuJoCo 3.10 nativo em
Apple Silicon. Verificado.

### ✅ MJ.2 — Cena do campo (`envs/mujoco/soccer_scene.py`)
Gera o MJCF do campo a partir do MJCF do robô (fonte única, robô intocado):
- Campo 14×9 m + margem, plano tipo grama, fricção com rolling friction calibrada
- Bola Size-5 (r=0.11 m, 0.43 kg), fricção 0.7/0.005/**0.005** (rola ~7 m e para a 3 m/s)
- Gols físicos (2 traves + travessão como capsules) em x=±7, largura 2.6 m
- Linhas do campo (visual, sem colisão)
- Helpers `is_goal()` / `is_out_of_bounds()` para a lógica de episódio
- **Nota física**: restituição do MuJoCo é limitada no timestep de treino (0.002 s) →
  quique ~6-8%. Aceitável: no futebol a bola quase sempre rola. O que importa
  (rolar/parar, colisão com trave) está validado.

### ✅ MJ.5 — Política base_walk no campo
`play_mujoco_soccer_walk.py` (viewer) + `scripts/render_walk_on_field.py` (headless MP4).
A política `base_walk_extended.pt` do htwk-gym anda no campo, empurra a bola e
detecta gol/out com reset só-da-bola. Validado headless: robô caminha 0.5 m/s,
colide com a bola e a empurra, permanece em pé (proj_grav_z = −1.00).

### Testes (`tests/unit`, 13 passando no Mac)
- Cena compila, robô intocado (12 atuadores), bola/gols/campo presentes
- Propriedades da bola (raio, massa)
- Física: quique não-degenerado, rola e para, trave bloqueia a bola
- Geometria: `is_goal` / `is_out_of_bounds`
- Política: fica em pé parado, anda pra frente >1 m em 8 s, curva sem cair

### Pendente (dependem do treino Isaac / GPU remota)
- MJ.3 `soccer_obs.py` compartilhado — será criado junto com o env Isaac (F1.3)
- MJ.4 wrapper env completo (49-dim obs) — idem
- MJ.6 harness de avaliação em grid — após primeiro checkpoint
- MJ.7 ponte de checkpoint Isaac→MuJoCo — após F1.6
