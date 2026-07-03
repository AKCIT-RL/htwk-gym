# Parte 1 — MVP: F0 → F1 → F2 → F4

Objetivo: política walk+kick **unificada** num campo completo, ainda sem AMP/encoder.
Cada etapa tem critério de validação explícito antes de avançar.

## Requisitos de hardware (IMPORTANTE)

Isaac Gym Preview **só roda em Linux x86_64 com GPU NVIDIA (CUDA)**. Não funciona em
macOS/Apple Silicon de jeito nenhum (nem via Docker/Rosetta — precisa de driver NVIDIA).

**Setup recomendado com MacBook:**
- **Desenvolvimento no Mac**: editar código, testes unitários puros (operadores de simetria,
  frames de referência, lógica de reward em torch-CPU), git.
- **Treino/simulação em máquina Linux+NVIDIA remota**: servidor do lab, ou cloud
  (vast.ai / Lambda / RunPod — uma RTX 4090 24 GB roda 4096 envs bem).
  Workflow: VS Code Remote-SSH ou git push → pull no servidor.
- Referência de custo: paper usou 8×V100/1 dia (16384 envs, 20k epochs).
  1×4090 com 4096 envs ≈ 3-6 dias para treino completo; treinos de validação
  curtos (500-3k iters) levam de minutos a poucas horas.
- Alternativa futura (não agora): portar para MuJoCo/MJX ou Isaac Lab. Custo de porte alto;
  manter Isaac Gym pela compatibilidade com o htwk-gym.

**Divisão prática dos testes:**
- `tests/unit/` — rodam no Mac (sem Isaac Gym): simetria, frames, potentials, shapes.
- `tests/sim/` — rodam só no servidor (importam isaacgym): episódio, ball-reset, contatos.

---

## F0 — Scaffold do projeto separado (~½ dia)

Pasta: `~/Documents/booster/tshingua/soccer-rl/`

```
soccer-rl/
├── docs/                     # estes MDs
├── envs/
│   ├── __init__.py
│   ├── base_task.py          # copiado do htwk-gym
│   └── T1/
│       ├── soccer.py         # NOSSA task (F1)
│       └── Soccer.yaml
├── utils/                    # copiados: runner.py, model.py, buffer.py, terrain.py, utils.py
├── resources/T1/             # URDFs + meshes + ball do htwk-gym
├── tests/
│   ├── unit/                 # rodam no Mac
│   └── sim/                  # rodam no servidor (isaacgym)
├── train.py / play.py
└── requirements.txt
```

Passos:
1. `git init`; copiar do htwk-gym: `utils/`, `envs/base_task.py`, `resources/T1/`,
   `train.py`, `play.py`, `requirements.txt`.
2. Limpar tasks antigas; loader dinâmico do runner aponta só p/ `envs/T1/soccer.py`
   (stub herdando de BaseTask).
3. Ambiente Python do servidor: Python 3.8 + torch compatível com Isaac Gym Preview
   (documentar no README; não usar uv/3.12 aqui).

**✅ Validação F0:**
- No servidor: `python train.py --task=Soccer --num_envs=16 --max_iterations=2` sem exception.
- No Mac: `pytest tests/unit` verde (imports básicos, sem isaacgym).

---

## F1 — Env `Soccer` (campo completo) (~1-2 semanas)

Base: `kicking_movement_bica.py` do htwk-gym. 7 sub-etapas testáveis:

### F1.1 — Campo e entidades
- Campo 14×9 m, plano com pequenas irregularidades (terrain existente).
- Bola Size 5: **raio 0.11 m** (o ball.urdf do htwk-gym tem 0.05 — criar novo URDF),
  massa ~0.43 kg, restituição/fricções randomizadas.
- Gols 2.6 m: região-alvo em coordenadas (x = ±7, |y| < 1.3), sem mesh/colisão por ora.
- Spawn: robô e bola uniformes no campo, yaw aleatório.

**✅ Validação:** inspeção visual no viewer (bola quica/rola, dimensões corretas);
teste: spawns dentro dos limites em 10k resets.

### F1.2 — Lógica de episódio
- 60 s (3000 steps @ 50 Hz). Termina APENAS em queda ou timeout.
- **Gol ou bola fora → reset SÓ da bola**; robô continua (ensina kicking contínuo).
- Eventos aleatórios: impulso/teleporte da bola (simula juiz/adversário); pushes no robô.

**✅ Validação:** com política aleatória: (a) episódio dura até queda/timeout;
(b) ball_resets > 0 sem reset do robô; (c) estado do robô idêntico antes/depois do
ball-reset. Logar `ball_resets/episode`.

### F1.3 — Espaço de observação
Actor (49 dims, com ruído):
| obs | dim |
|---|---|
| gravidade projetada | 3 |
| vel. angular base | 3 |
| dof_pos − default | 12 |
| dof_vel | 12 |
| ação anterior | 12 |
| ball pos (x,y) frame robô | 2 |
| ball mask | 1 |
| goal pos (x,y) frame robô | 2 |
| goal dir (cosθ,sinθ) frame robô | 2 |

- Nesta fase: ball mask = 1 sempre; ball pos = verdade + ruído gaussiano fixo.
  **Isolar em `_get_ball_observation()`** para trocar pela virtual perception na F6.
- Critic (privilegiado, +12): lin vel (3), base height (1), mass rand (4),
  ball vel xy mundo (2), ball friction (2).

**✅ Validação:** teste de frames — bola em (1,0) mundo, robô yaw=90° → obs (0,−1) no
frame do robô (idem goal dir). Shapes e ausência de NaN em 1k steps.

### F1.4 — Rewards (Tabela 3, sem AMP/cabeça)
| reward | peso | implementação |
|---|---|---|
| survival | +3 | constante/step |
| termination | −1000 | na queda |
| stagnation | −100 | deslocamento ~0 por 1 s |
| goal scored | +15 | bola cruza linha do gol |
| ball approach | +50 | potential-based: Φ=−‖robô−bola‖, r=γΦ'−Φ |
| goal progress | +500 | potential-based: Φ=−‖bola−gol‖ |
| sideways kick | +20 | vel. LATERAL do pé (frame do pé) durante contato pé-bola |
| forward kick | −20 | vel. frontal do pé durante contato |
| foot proximity | −5 | pés < d_min |
| leg action rate | −1 | ‖a_t−a_{t−1}‖² |
| joint pos limit | −100 | violação |
| base acceleration | −0.001 | |
| collision | −100 | contato corpo ≠ pé |

Atenções:
- Potential-based: **mascarar o step do ball-reset** (não computar salto de potencial).
- Kick de chapa: sensor de contato pé-bola + velocidade do pé no frame local do pé.

**✅ Validação:** (a) testes dos potentials (aproximar bola → r>0; ball-reset → r=0 no
step); (b) treino 500 iters: `ball approach` sobe primeiro, `goal progress` depois;
(c) assistir rollouts (reward hacking check).

### F1.5 — Randomização e robustez
- Reaproveitar: massa/CoM, PD gains, fricção dos pés, pushes.
- Adicionar: propriedades da bola por episódio.

**✅ Validação:** histograma dos sorteios; treino não diverge com randomização ligada.

### F1.6 — Treino baseline (critic único)
- 4096 envs, 3-5k iters. Expectativa: robô se aproxima da bola e a empurra ao gol,
  desajeitado. É baseline, não produto.

**✅ Critérios de aceite F1:**
- `goal scored` > 0 e crescendo; quedas < 20% dos episódios;
- vídeo: robô navega até a bola de qualquer pose inicial;
- benchmark registrado (régua para F2/F4).

### F1.7 — Harness de avaliação (grid da Fig. 3A)
- `evaluate.py`: bola em grid de posições fixas, robô no centro, N trials/célula;
  medir taxa de gol, quedas, tempo até contato. Headless.

**✅ Validação:** gera heatmap tipo Fig. 3A. Métrica oficial do projeto daqui em diante.

---

## F2 — Multi-critic PPO (~2-3 dias)

1. `utils/model.py`: 2 cabeças de valor — `critic_goal`, `critic_aux` (256,256,128).
2. Env: separar `rew_goal` (goal scored + ball approach + goal progress) e `rew_aux` (resto).
3. `buffer.py`/`runner.py`: 2 GAEs independentes (γ=0.995, λ=0.95), advantages
   normalizados separadamente, `A = 2·Â_goal + 1·Â_aux`. Value loss = soma das MSEs.

**✅ Validação:**
- Regressão: com `rew_aux=0`, idêntico ao critic único.
- A/B mesma seed, 3k iters: multi-critic deve ser melhor/mais estável (Fig. 4A do paper).
  Se não reproduzir, investigar antes de seguir.
- Explained variance dos 2 críticos > 0.

---

## F4 — Symmetry loss (~2-3 dias, paralelo à F2)

1. Operadores de espelhamento do T1 (derivar formalmente — fonte nº 1 de bugs):
   - `M_obs`: troca juntas esq↔dir; nega roll/yaw das juntas; reflexão no plano XZ:
     nega y de vetores (gravidade, ball/goal pos) e nega x,z de pseudo-vetores (ang vel);
     nega sinθ do goal dir.
   - `M_act`: troca esq↔dir, nega roll/yaw.
2. Loss: `L_sym = ‖μ(o) − M_act(μ(M_obs(o)))‖²`, coef 10, no update do PPO.

**✅ Validação:**
- Unitário: `M(M(x)) == x` (involução).
- Físico: ação espelhada em estado espelhado → trajetória espelhada por alguns steps
  no simulador (pega erro de sinal em pseudo-vetor).
- Pós-treino: chutes esquerda/direita ~50/50 no harness (sem a loss degenera p/ 90/10).

---

## Cronograma e marcos

```
Semana 1:   F0 + F1.1–F1.3
Semana 2:   F1.4–F1.5 + testes
Semana 2-3: F1.6 (baseline) + F1.7 (harness)      ← MARCO 1: baseline funcional
Semana 3:   F2 ∥ F4 + treinos A/B
Semana 4:   treino MVP completo (f1+f2+f4)         ← MARCO 2: MVP walk+kick unificado
```

**Definition of Done da Parte 1:** no harness, gol > 50% nas regiões próximas;
quedas < 10%; chute bilateral ~50/50; vídeo com transição andar→chutar sem pausa
(movimento ainda "robótico" — fluidez humana vem do AMP na Parte 2).

---

## Anexo: análise de simuladores (decisão registrada em 2026-07-03)

Pergunta: dá pra rodar num "Isaac" no Mac ou x86 sem NVIDIA? **Não.**
- Isaac Gym Preview: Linux x86_64 + GPU NVIDIA obrigatória (física em CUDA).
- Isaac Sim / Isaac Lab: exige GPU NVIDIA RTX. Sem versão macOS.

Alternativa Mac-native: **MuJoCo / MuJoCo Playground** (citado no próprio paper, ref [45];
já inclui o Booster T1 oficialmente).
- ✅ Roda nativo em Apple Silicon: viewer, debug, poucos envs.
- ❌ Treino em massa usa MJX (JAX) → precisa CUDA/TPU; Metal backend imatura;
  Mac do time tem 8 GB RAM.
- ❌ Porte p/ JAX: reimplementar AMP/multi-critic/encoder-decoder, perde reuso do htwk-gym.

**Decisão: Rota A — Isaac Gym + GPU NVIDIA remota.**
- Mac: desenvolvimento, testes unitários (tests/unit), análise.
- Servidor/cloud (RTX 4090 ~US$0.3-0.5/h em vast.ai/RunPod): tests/sim + treinos.
- Bônus: o pipeline Isaac→MuJoCo do htwk-gym (play_mujoco_*.py) permite VISUALIZAR
  políticas treinadas no Mac via MuJoCo — igual ao demo da Tsinghua no notebook.
- Revisitar MJX/Playground apenas se a infra NVIDIA se tornar um gargalo real.
