# Implementação do paper arXiv:2511.03996 — "Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots"

Tsinghua/ByteDance — campeão RoboCup 2025 Adult-size (Tsinghua Hephaestus, 76 gols / 11 sofridos).
Robô: Booster T1 (mesmo do nosso htwk-gym). Project page: https://humanoid-kick.github.io

## A ideia central
Uma **única política** (PPO + asymmetric actor-critic, Isaac Gym) que recebe percepção
visual processada (bola + gol no frame do robô) e proprioceção, e produz posições de junta
a 50 Hz — incluindo as 2 juntas do pescoço (percepção ativa). Sem segmentação manual
walk/kick: os comportamentos (andar, girar, chutar L/R, procurar bola) **emergem** numa rede só.

### Os 4 ingredientes da fluidez
1. **AMP (WGAN-GP)** — discriminador com ~76 s de caminhada (ACCAD, público) + ~30 s de
   mocap de chute de chapa, retargeted pro T1. Reward de estilo peso 0.3.
2. **Encoder-decoder** — 50 frames de história (1 s) → latente 64-dim; decoder reconstrói
   estados privilegiados (posição verdadeira da bola). Denoising: RMSE 0.344 m → 0.186 m.
3. **Multi-critic** — critic de gol (goal scored/ball approach/goal progress) e critic
   auxiliar (estilo/regularização). A_total = 2·A_goal + 1·A_aux.
4. **Virtual perception system** — em sim, a bola é observada com ruído/latência/FOV/taxa
   medidos no robô real: ruído N(0,(0.124d+0.149)²), latência N(116ms,18ms²),
   freq N(25.36Hz,1.06Hz²), detecção 90% até 7 m.

## Especificação técnica (Tabelas 1–3 + apêndices do paper)

### Arquitetura
- Actor MLP (256,256,128) ELU; Critic (256,256,128); Encoder (1024,128) → 64; Decoder (128,128)
- Ação: posições de junta (PD) @ 50 Hz, pernas 12 DOF + pescoço 2 DOF

### Observações
- **Actor**: gravidade projetada, ang vel, dof pos/vel, ação anterior, ball pos (x,y) frame
  robô, ball mask, goal pos (x,y), goal dir (cosθ,sinθ)
- **Critic (privilegiado)**: + lin vel, base height, mass rand, ball vel, ball friction
- **Decoder reconstrói**: ball position verdadeira (+ dinâmica)

### Rewards (Tabela 3)
| Reward | Peso |
|---|---|
| Survival | +3 |
| Termination (queda) | −1000 |
| Stagnation (parado 1 s) | −100 |
| Goal scored | +15 |
| Ball approach (potential-based) | +50 |
| Goal progress (potential-based) | +500 |
| Head pitch/yaw alignment | −0.5 cada |
| AMP style | +0.3 |
| Sideways kick (contato) | +20 |
| Forward kick (contato) | −20 |
| Foot proximity | −5 |
| Head action rate | −15 |
| Leg action rate | −1 |
| Joint position limit | −100 |
| Base acceleration | −0.001 |
| Collision | −100 |

### Hiperparâmetros (Tabela 1)
γ=0.995, GAE λ=0.95, KL alvo 0.01, entropy 0.01, lr adaptativo, 5 epochs, minibatch 4,
coef reconstrução 1, discriminador 1, gradient penalty 50, symmetry 10.
Treino do paper: 16384 envs, 20k epochs, 8×V100, ~1 dia.

### AMP (WGAN-GP, apêndice C)
- L_D = −E[tanh(0.4·D(x_E))] + E[tanh(0.4·D(x_π))], sobre transições (s_t, s_{t+1})
- Gradient penalty coef 50; r_amp = −tanh(0.4·D(x_π))
- Reference State Initialization (RSI); mirror symmetry loss coef 10

### Ambiente
Campo 14×9 m, gols 2.6 m, bola Size 5 randomizada; episódio 60 s; termina só em queda;
gol/bola fora → reset apenas da bola; teleporte/impulso aleatório da bola; pushes no robô;
terreno levemente irregular.

## Roadmap (todos rastreados no banco da sessão)
```
FRENTE A (RL/env — caminho crítico)
  f0-scaffold → f1-env-soccer → f2-multi-critic → f3-encoder-decoder
                             └→ f4-symmetry-loss
                             └→ f6-virtual-perception → f7-neck-control
FRENTE B (dados)
  f5a-amp-dataset → f5b-amp-discriminator
FRENTE C (percepção real, paralela)
  f8-real-perception (YOLOv8 + BEV + odometria)
CONVERGÊNCIA
  f9-training-runs → f10-sim2sim-deploy
```

**MVP (Parte 1)** = f0+f1+f2+f4 → política walk+kick unificada "robótica" mas funcional.
Detalhes em `01-parte1-mvp.md`.

## Decisões em aberto
1. Demos de chute (f5a): vídeo-to-motion (GVHMR/WHAM) vs keyframes vs política atual como pseudo-demo
2. Compute: GPU NVIDIA remota necessária (ver `01-parte1-mvp.md` § Requisitos de hardware)
3. Ground truth p/ calibrar virtual perception (f6): mocap disponível? Senão, params do paper
4. ✅ URDF T1 tem Head_yaw/Head_pitch (T1_serial.urdf, 23 juntas) — f7 viável
