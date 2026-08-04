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
- **2026-08-01 (3)** — Frentes **C e D implementadas e validadas** (usuário adiantou
  a D). C: randomização com ranges HTWK deploy-proven — bola (massa ×[0.7,1.3],
  fricção [0.2,1.2], restituição [0.1,0.9]), robô (fricção [0.1,2.0], massa base
  ×[0.8,1.2], CoM ±0.1 m, demais massas ×[0.98,1.02]), empurrões (vel std 0.3 m/s
  a cada 5–10 s) e terreno irregular ±2 cm em **tiles por campo** (borda achatada
  em z=0). D: `steer_anneal_start/end_samples` escala linearmente a obs de steering
  1→0 (config: 5M→15M). Lição Isaac Gym: trimesh global + `env_spacing>0` quebra o
  broadphase GPU (colisão dependia do env, não da posição mundial); solução =
  `env_spacing 0` + atores criados já espalhados nos campos (empilhar na origem no
  build estoura `foundLostAggregatePairsCapacity` a 2048 envs). Gates:
  `validate_soccer_randomization.py` 16/16, regressão E2–E4 OK, 65/65 CPU, smoke
  2048 envs limpo. Commits: MimicKit `0508035`, htwk-gym `56707d7`. Treino C
  lançado (`soccer-smoke-c`, wandb `zwjhp584`, 20M, warm-start P1+P2); D treina
  em seguida (warm-start P4b@it200; eval com `mcwamp_g1_soccer_env_d_eval.yaml`,
  steering zerado — pergunta é "remover a muleta sem colapso", não bater P4b).
- **2026-08-01 (4)** — **Treino C concluído e avaliado** (~1h25, 20M samples).
  Env padrão sem perturbação (vs P4b@it200 1.61 gols/ep, 6.2% quedas, 70.4°):
  C@it305 = **2.11 gols/ep, 3.1% quedas, 59.1° erro angular, 13.2 chutes/ep** —
  a randomização melhorou até o desempenho nominal. Sob perturbação completa
  (env C: terreno+rand+push), quedas **26.5% vs 41.8%** do P4b (−37% rel.),
  gols 0.60 vs 0.51, episódios 50.1s vs 42.4s. **Novo melhor ckpt:
  `output/mcwamp_g1_soccer_smoke_c_seed1/int_models/model_0000000305.pt`.**
  Treino D lançado (`soccer-smoke-d`, warm-start P4b@it200, anneal 5M→15M).
- **2026-08-01 (5)** — **Treino D concluído e avaliado** (steering zerado,
  `mcwamp_g1_soccer_env_d_eval.yaml`). Controle P4b@it200 sem steering:
  **colapso total** — 100% quedas, 0 gols, episódios de 13.9s (a muleta era
  estrutural para o equilíbrio). D@it305 sem steering: **9.4% quedas, episódios
  de 56.4s** — o anneal removeu a dependência física com sucesso. Porém a tarefa
  degradou: 0.06 gols/ep, 1.75 chutes/ep — a política fica estável mas passiva;
  o steering também carregava o comportamento "ir até a bola". Veredito parcial:
  muleta removível sem colapso físico, mas o engajamento na tarefa precisa ser
  reaprendido. Próximo passo recomendado (D2): anneal a partir do **C@it305**
  (política mais forte) com mais orçamento pós-anneal (ex.: 30M, anneal 5M→15M,
  15M finais com steering zerado), antes de concluir a frente.
- **2026-08-01 (6)** — **D2 reprovado no gate; D3 quase lá; D4a no ar.**
  D2 (C@305 + anneal 5M→15M, 30M): engajamento morre **durante** o fade
  (@13M: 0.78 chutes/ep) e nunca volta (final: 1.02 chutes/ep, 0.02 gols);
  pior até com steering (0.63 gols vs 2.11 do C@305). Recompensa auditada: não
  depende do steering — a política usava a obs como sinal interno de "andar" e
  cai no **atrator passivo** (PPO on-policy não redescobre navegação). Veredito:
  anneal gradual NÃO funciona. D3 (command-free desde o passo 0, warm-start
  P1+P2, env C, 30M): **0.50 gols/ep, 3.9 chutes/ep, 1.6% quedas, 61.6°** sem
  nunca ver comando — confirma que evitar a formação da dependência funciona,
  mas engajamento fraco/bimodal (±5.4). D4 (sugestão do usuário): currículo de
  approach — D4a overfita "andar até a bola" (w 50→250, 10M, command-free) e
  D4b restaura w=50 (20M, warm-start D4a). D4a rodando (`soccer-smoke-d4a`).
- **2026-08-01 (7)** — **D4 reprovado; D5 (continuação do D3) no ar.** D4a fez
  o esperado: navegação excelente (engage latency 0.79s vs 2.26s do D3) mas
  chute suprimido (1.03 chutes/ep, 115°) — com w=250, afastar a bola custa caro.
  D4b (w restaurado a 50, 20M): supressão NÃO reverteu (1.09 chutes/ep, 0.09
  gols, 0% quedas) — mesma histerese do D2: comportamento suprimido numa fase
  não volta com PPO on-policy. **Lição consolidada: currículos de duas fases
  (anneal ou reward) criam hábitos irreversíveis; treinar direto no regime
  final funciona melhor.** Curva do D3 medida: 13M→30M = 0.06→0.50 gols/ep,
  1.1→3.9 chutes/ep — longe do platô. D5 = D3 + 30M extras (mesma config,
  variável única = orçamento), rodando (`soccer-smoke-d5`).
- **2026-08-02** — **Frente D ESTACIONADA por decisão do usuário** (após
  inspeção visual no viewer). D5 interrompido @9M da extensão (~39M totais):
  3.75 chutes/ep, 0.42 gols/ep — igual ao D3, ganho marginal. Diagnóstico
  visual+métrico do porquê o robô "orbita" a bola sem chutar no regime
  command-free: (i) approach reward potential-based zera ao chegar perto —
  sem gradiente de "agora chute"; (ii) rewards de chute só pagam após a bola
  se mover; (iii) **causa raiz: prior AMP só tem walk/run — armar chute é
  OOD/punido pelo discriminador** (mesmo teto de ~60-70° em todas as
  variantes). Conclusões da frente: muleta É removível sem colapso (D3
  navega e marca command-free), mas fechar o gate exige a Frente A (dataset
  de chute). Ckpt command-free de referência: D5 model.pt (~39M,
  `output/mcwamp_g1_soccer_smoke_d5_seed1/model.pt`). Melhor geral segue
  C@it305. Próximo: H (avaliação estilo paper) e E (percepção virtual),
  conforme §4.
- **2026-08-03** — **Frente E implementada e gated; treino direto REPROVADO
  (requer Frente F).** Pipeline de câmera virtual (§9 do paper) em
  `task_soccer_env.py`: ruído σ=0.124d+0.149, latência N(116,18²) ms com
  **ring buffer K=8** (latência > período ⇒ vários frames em voo; slot único
  nunca entrega — bug pego no gate), frequência N(25.36,1.06²) Hz por
  episódio, detecção 90%≤7 m decaindo até 10 m, FOV 120° no heading,
  zero-order hold + máscara real na obs (contrato 249 dims intacto; rewards
  usam estado verdadeiro). Gates: 12/12 GPU
  (`validate_soccer_perception.py`; ruído medido 0.522 vs 0.521 esperado),
  69/69 CPU, regressão C+D 16/16 e E2-E4 OK. **Treino E (20M, warm-start
  P1+P2, protocolo idêntico ao C): política colapsa** — E@305 cai 67% no env
  E e 77% no env limpo; E@200 igual (64%/62% — não é divergência tardia);
  C@305 sob percepção cai 100%. Train_Return E -726 vs C +203. Diagnóstico:
  política **sem memória** não filtra ruído de ~0.5-0.8 m a 25 Hz + latência
  116 ms — o paper resolve com encoder temporal/história (nossa Frente F).
  Conclusão: infra E validada e commitada (MimicKit 9972ccc); treinar com
  percepção ligada fica bloqueado até a Frente F (ou história de obs).
  1º lançamento do treino morreu na it0 (exit 1 transitório); repro rodou
  limpo. Artefatos: `output/soccer_e_diag/`.
- **2026-08-03** — **Frente H implementada e VALIDADA.**
  `evaluate_soccer_policy.py` estendido: grade 3×3 de taxa de gol por região
  do posicionamento da bola (x orientado ao gol) + time-to-first-kick por bin
  de ângulo de aproximação (0/45/90/135/180°). Regressão: C@305 no env padrão
  reproduz exatamente o baseline (2.109 gols/ep, 3.1% quedas). Números H do
  C@305 (env C uneven, no_perturb): 9.8% dos 459 posicionamentos viram gol;
  melhor célula = meio-campo longe do gol (25.4%); time-to-kick cresce com o
  ângulo (3.8 s @45-90° → 5.1 s @135-180°). Melhor geral segue C@it305.
  Próximo: **Frente A (dataset de chute — usuário produzindo em paralelo)**;
  Frente F sobe de prioridade (desbloqueia E).
- **2026-08-03 (2)** — **Frente F-lite implementada; F1 APROVADO; F2 reprovado;
  F3 (critic assimétrico) implementado e no ar.**
  - **Infra F:** `task_obs_history_steps` (default 0 = contrato 249 intacto)
    anexa H×12 dims de história do bloco de tarefa (steer 5 + soccer 6 + mask 1)
    → 369 dims com H=10. Roll único por passo em `_update_task`, refill por
    broadcast em `_reset_envs`, `_compute_obs` sem mutação (é usada como probe
    de shape por `get_obs_space`). Warm-start por expansão zero-init do ckpt
    (`expand_checkpoint_history.py`: 120 colunas zeradas em actor/critic/aux +
    obs_norm mean 0/std 1) — equivalência exata provada no gate. Gates:
    `validate_soccer_history.py` 10/10 + regressão 249 (2 processos — Isaac Gym
    só permite 1 sim/processo), percepção 12/12, CPU 69/69.
  - **F1 (env C + hist10, sem percepção, 20M):** história é não-inferior e até
    superior — par justo plane (env com pushes): **1.64 vs 1.27 gols/ep** do
    C@305, 67.8° vs 70.7°; uneven: **0.72/19.1%/9.3** vs 0.59/25%/8.6.
    Nota: o baseline "2.11/3.1%" do C@305 era no yaml padrão sem pushes;
    Train_Return negativo é normal (C@305 = −690; usar Test_Return: 435 vs 203).
  - **F2 (env E + hist10, 20M): REPROVADO** — 64.1% quedas (73.4% no env
    limpo), 0.20 gols, 1.3 chutes: colapso global igual ao E. Janela de 10
    frames no ator NÃO basta; o critic compartilhado vendo a bola
    ruidosa/latente envenena o valor globalmente.
  - **F3 (critic assimétrico privilegiado, Table 2 do paper):** env publica
    `info["critic_obs"]` quando `virtual_perception=true` (layout do ator com
    bola VERDADEIRA no bloco + história privilegiada, mask=1); MCWAMP roteia os
    critics goal+aux para `critic_obs` com normalizador próprio (`_critic_obs_norm`),
    ator intacto; load tolerante (seed do `_obs_norm` para ckpts antigos, drop
    das chaves em env sem percepção). Gate `validate_soccer_asym_critic.py`
    13/13 (inclui 1 iteração de treino end-to-end). **Resultado F3@305:**
    quedas **18.8%** no env com percepção (E/F2: 64-67%) e 21.5% no limpo —
    envenenamento global ELIMINADO, quedas no nível do F1 (19.1%); 0.32 gols,
    5.1 chutes. Test_Return +507 vs −623 do F2. 18.8% ∈ [10%,25%] → gate manda
    **estender orçamento**: F3b (+20M a partir do F3 model.pt) rodando.
    Aprova <10% quedas; se estagnar >25%, próximo degrau = encoder-decoder
    completo do paper.
- **2026-08-03 (3)** — **F3b REPROVADO; Frente F encerrada com F3@305 como
  referência sob percepção.** A extensão +20M derrete o engajamento
  monotonicamente (chutes/ep 5.1 → 1.9 @200 → 1.2 @305; gols 0.32 → 0.06)
  enquanto as quedas melhoram (18.8% → 10.6%): é o **atrator passivo** da
  Frente D reaparecendo — sob bola ruidosa/latente, "não chutar" é ótimo
  local (o prior AMP não tem chute e os rewards de chute pagam pouco vs o
  risco). Mais orçamento só aprofunda o hábito (consistente com a lição de
  D2/D4: PPO on-policy não reverte comportamento suprimido). Conclusões da
  Frente F: (i) história de task-obs é não-inferior e vira ganho no env limpo
  (F1 > C); (ii) critic assimétrico é NECESSÁRIO para treinar sob percepção
  (F3 elimina o colapso 64→19% de quedas); (iii) o teto de engajamento sob
  percepção é o mesmo teto da Frente D — a causa raiz é o motion prior sem
  chute, não a percepção. **Refs:** sem percepção = F1@305
  (`output/mcwamp_g1_soccer_smoke_f1_seed1/int_models/model_0000000305.pt`);
  sob percepção = F3@305
  (`output/mcwamp_g1_soccer_smoke_f3_seed1/int_models/model_0000000305.pt`).
  Encoder-decoder do paper fica para depois da **Frente A** (dataset de
  chute), que ataca a causa raiz. Próximo: Frente A.
