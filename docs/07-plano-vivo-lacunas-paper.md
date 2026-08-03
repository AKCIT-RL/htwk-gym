# Plano vivo — lacunas vs paper (arXiv:2511.03996)

> **Documento vivo.** Atualizado pelo agente a cada marco fechado ou decisão nova.
> Fonte: paper HTML v1 (relido em 2026-08-01, seções 4-10 + Tabelas 1/2) + código do workspace.
> Última atualização: **2026-08-01** (pós-ablações P4/P4-near/P4b).

## 1. Resposta à pergunta central

**Não precisamos do dataset de chute para continuar.** Das ~12 frentes do paper, só UMA
(qualidade do chute: direção/força/contato com o arco do pé) depende do dataset de
motions de chute. Todas as demais são implementáveis já. Porém, as ablações P4
provaram empiricamente que **reward tuning não melhora mais o chute** (erro angular
estagnado em ~70°, impacto ~1.5 m/s) — o teto é o motion prior (AMP só tem walk/run).
Logo: as outras frentes melhoram robustez, deployabilidade e mecânica de treino,
mas o ganho de futebol propriamente dito está bloqueado no dataset.

## 2. Estado atual (o que JÁ temos)

| Componente do paper | Status | Evidência |
|---|---|---|
| Campo RoboCup Adult 14×9 m, gol 2.6 m | ✅ | task_soccer_env.py, gate GPU 32/32 |
| Soft reset da bola (gol/OOB), robô continua | ✅ | validado; **near** venceu A/B (far colapsa chute) |
| Perturbações da bola (teleporte/push, "árbitro/oponente") | ✅ | `_update_ball_perturb`, 4-8 s |
| Rewards densos potential-based (approach, goal progress) | ✅ | soccer_util.py |
| Multi-critic (goal vs aux), advantage w_goal=2/w_aux=1 | ✅ | `critic_weights: [2.0, 1.0]` — IGUAL ao paper |
| AMP + WGAN (tanh(0.4·D), GP=50) | ✅ | wamp_agent.py, wgan_util.py, `disc_grad_penalty: 50` |
| RSI (reset amostrando dos clips) | ✅ | MimicKit padrão (`rand_reset`) |
| Kick rewards T1-style (direction, gating, waiting) | ✅ ajustado | P4b: min_vel 0.3, near reset (commits bc363d2/5c94114) |
| Steering warm-start (MULETA nossa, não existe no paper) | ⚠️ temporário | obs 237:242; precisa anneal (frente D) |
| Harness de avaliação (gols, OOB, kicks, impacto, ângulo) | ✅ | scripts/evaluate_soccer_policy.py |
| Viewer com campo e setas de debug (opt-in) | ✅ | flags default false, commit c0fcbef |

**Melhor checkpoint:** `output/mcwamp_g1_soccer_smoke_p4b_seed1/int_models/model_0000000200.pt`
(1.61 gols/ep, 6.2% quedas, 68.8% goal rate; 20M samples). Baseline para próximos A/Bs.

## 3. Lacunas (o que FALTA), por frente

### Frente A — Dataset de motions de chute (ÚNICO bloqueio do chute)
**Paper:** 76.28 s de caminhada omnidirecional do **ACCAD** (público!) + **30 s de chute
com arco do pé** capturados por mocap próprio (privado). Retarget para o robô.
**Nosso estado:** só g1_walk + g1_run (1.83 s total!) — dataset ordens de grandeza menor.
**Caminhos sem mocap próprio (decisão do usuário pendente):**
1. ACCAD/AMASS público: clips de caminhada omnidirecional (igual ao paper) + clips de
   chute públicos (CMU/AMASS têm "kick soccer ball") → retarget via
   `MimicKit/tools/gmr_to_mimickit/` ou `tools/smpl_to_mimickit/` (JÁ EXISTEM no repo).
2. Keyframe/procedural: animar chute com arco do pé à mão no frame do G1 (mais rápido,
   menos natural).
3. Mocap próprio (fiel ao paper, mais caro).
**Critério de saída:** dataset ≥ 60 s walking omni + ≥ 20 s chute; view_motion OK;
disc não colapsa; erro angular do chute < 45° e impacto > 2.5 m/s em 20M A/B.

### Frente B — Mecânica de treino (Tabela 1) — TRILHA DO COLEGA (S2/S3)
| Item | Paper | Nosso | Gap |
|---|---|---|---|
| Learning rate | adaptativa, desired KL 0.01 | fixa | S2 |
| Discount | 0.995 | 0.99 | S2 |
| Entropy coef | 0.01 | 0.0 | S2 (cuidado: std fixa → gradiente zero, ver briefing) |
| Mirror/symmetry loss | coef 10, ℒ=‖a−M_a(π(M_o(o)))‖² | ausente | S3 — mata o viés 82% pé esquerdo SEM dataset novo |
| Épocas de aprendizado | 5 | conferir | S2 |
| Ativação | ELU | ReLU | baixa prioridade (quebra warm-start!) |
| Escala | 16384 envs, 20k epochs, ~1 dia 8×V100 | 2048 envs, 20M smoke | esperado; full run só pós-contrato final |
**Nota:** a symmetry loss é a frente com melhor razão ganho/custo AGORA — ataca
diretamente o viés de pé (82% esq) e a paper diz que é "critical" para chute bilateral.

### Frente C — Randomização de ambiente (robustez)
| Item | Paper | Nosso |
|---|---|---|
| Propriedades da bola randomizadas (massa, atrito, restituição) | ✅ | ❌ |
| Terreno levemente irregular | ✅ | ❌ (flat) |
| Empurrões/velocidades extras no robô ("confronto físico") | ✅ | ❌ (só na bola) |
| Randomização de massa/CoM do tronco (Tabela 2: obs do critic) | ✅ | ❌ |
**Sem dependência do dataset.** Custo baixo, ganho de robustez; pré-requisito do sim-to-real.

### Frente D — Anneal da muleta de steering (P1)
O ator do paper NÃO recebe comando de steering — aprende busca/aproximação sozinho.
Nossa política depende dos slots 237:242. Plano: decair `steer_speed_max`→0 ao longo
do treino ou zerar os slots com probabilidade crescente; critério: métricas do harness
estáveis com steering zerado.

### Frente E — Percepção virtual (parâmetros EXATOS no paper, §9)
- Ruído posicional: 𝒩(0, (0.124·d + 0.149)²), d = distância à bola.
- Latência: 𝒩(116 ms, (18 ms)²). Frequência: 𝒩(25.36 Hz, (1.06 Hz)²).
- Detecção: 90% dentro do FOV até 7 m, decaindo além.
- Máscara de bola REAL (hoje é sempre 1; política nunca viu oclusão).
**Sem dependência do dataset.** Importante: E **não quebra o contrato** — os slots da
bola/máscara já existem no layout; só passamos a aplicar ruído/latência/dropout neles.
Pode (e vai) ser feita ANTES da Frente F.

### Frente F — Encoder-decoder + obs mensuráveis (QUEBRA DE CONTRATO — fazer 1 vez)
- Ator (Tabela 2): gravidade projetada, vel. angular (IMU), offset de juntas, vel. de
  juntas, ação anterior, bola (x,y), máscara, gol (x,y), direção do gol (cos,sin).
  **SEM** vel. linear, altura da base, key bodies (nosso char obs 237 é privilegiado).
- Critic (AAC): estado completo + vel. da bola + mass randomization etc.
- Encoder: 50 frames (1 s) de histórico → latente 64-d, concatenado à obs atual.
- Decoder: reconstrói estados privilegiados (posição real da bola, dinâmica) — treinado
  JUNTO (coef 1); pós-hoc fica no nível do ruído (comprovado no paper, Fig. 4B).
- Perde warm-start P1+P2 e todos os baselines → juntar E+F+máscara real+randomização
  num único retreino (decisão já ratificada pelo usuário: "não mudar nada relacionado
  a deploy por enquanto").

### Frente G — Cabeça ativa e FOV
Paper: T1 com câmera em 2 DOFs (yaw/pitch) de pescoço; tracking ativo emergente; reward
de manter bola no centro do FOV (opcional, mas melhora robustez).
**G1 29DOF NÃO tem pescoço atuado** → adaptação: FOV fixo no torso ou yaw da cintura
como proxy. Decisão de projeto pendente; só relevante junto com E/F.

### Frente H — Avaliação estilo paper (sem dependências)
- Grade de success rate por região do campo (paper: 8192 testes; nosso: reduzido).
- UMAP dos clusters de gait (walk, turn L/R, kick L/R) — mede diversidade.
- Tempo até o chute vs ângulo de aproximação (comparação com baseline rule-based).
Extensões do harness atual; implementável já.

### Frente I — Odometria e deploy (pós-tudo)
MLP proprioceptivo autoregressivo (1 s de histórico) + particle filter com landmarks.
**Deploy-only** — não afeta treino. Ver docs/06 (gates G0-G8). Congelado por decisão
do usuário (2026-08-01): "não vamos mudar nada relacionado".

## 4. Ordem e divisão de responsabilidades (decisão do usuário, 2026-08-01)

**Colega:** Frente B inteira (mirror loss S3 + hiperparâmetros S2).

**Nós, agora:**
1. **C** — randomização de bola/terreno/pushes no robô (3 A/Bs de 20M, 1 variável cada,
   vs baseline P4b@it200).
2. **E** — percepção virtual SEM quebra de contrato (ruído, latência, frequência,
   máscara real de oclusão nos slots existentes da bola).

**Depois ("o resto a gente testa depois"):**
3. **H** — avaliação estilo paper (barato, melhora todos os A/Bs futuros).
4. **D** — anneal do steering.
5. **A** — dataset de chute quando o usuário decidir a fonte (ACCAD/AMASS = menor atrito;
   ferramentas de retarget já existem no repo).
6. **F(+G)** — quebra única de contrato: encoder-decoder + obs mensuráveis. Só depois
   de A (para não retreinar duas vezes o contrato novo).
7. **I** — deploy (docs/06, congelado).

## 5. Decisões pendentes do usuário

- [ ] Fonte do dataset de chute (ACCAD/AMASS público vs keyframe vs mocap próprio).
- [ ] Adaptação da cabeça no G1 (sem pescoço): FOV fixo vs yaw de cintura como proxy.
- [ ] Quando pagar a quebra de contrato E+F (sugestão: depois de A validado).
- [ ] Escala do run final (paper: 16384 envs / ~1 dia em 8×V100; nós: 1 GPU).

## 6. Log de atualizações

- **2026-08-01** — Criação. Pós-P4: near reset + min_vel 0.3 promovidos (P4b@it200 =
  novo baseline: 1.61 gols/ep, 6.2% quedas, erro angular 70.4°). Descoberta-chave:
  reward tuning esgotado; teto é o motion prior. Paper relido (v1 HTML): parâmetros
  exatos de percepção (§9), Tabelas 1/2, WGAN tanh(0.4·D)+GP50 (já igual), multi-critic
  w=[2,1] (já igual), mirror loss coef 10 (falta), ACCAD público como fonte de walking.
- **2026-08-01 (2)** — Divisão de trabalho ratificada pelo usuário: colega faz B;
  nós fazemos C e E agora; D/F/G/H/A/I ficam para depois. Constatado que E sozinha
  não quebra o contrato de obs (slots da bola já existem).
