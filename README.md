# soccer-rl

Implementação do paper **arXiv:2511.03996** — *"Learning Vision-Driven Reactive
Soccer Skills for Humanoid Robots"* (Tsinghua/ByteDance, campeão RoboCup 2025
Adult-size). Alvo: robô Booster T1, uma única política RL walk+kick unificada.

Ver `docs/` para a engenharia de requisitos e o roadmap completo.

## Estado atual
- ✅ **F0 scaffold** + **MJ.2/MJ.5** — ambiente de campo no MuJoCo (roda no Mac),
  validado com a política `base_walk` pré-treinada do htwk-gym.
- ⏳ Próximo: env `Soccer` no Isaac Gym (treino em GPU NVIDIA remota).

## Setup (Mac / dev)
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Rodar
```bash
# Viewer interativo (macOS PRECISA de mjpython, nao python!):
./run_viewer.sh --command 0.4,0,0
#   ou:  .venv/bin/mjpython play_mujoco_soccer_walk.py --command 0.4,0,0
#   Teclas (clique na janela do MuJoCo primeiro):
#   W/S vx  A/D wz  Q/E vy  Space stop  R reset  B bola aleatória  K chute-fake  P estado

# Render headless (MP4, sem GUI)
.venv/bin/python scripts/render_walk_on_field.py --command 0.5,0,0 --seconds 5

# Testes (rodam no Mac, sem GPU)
.venv/bin/python -m pytest tests/unit -q
```

## Estrutura
```
envs/mujoco/soccer_scene.py     # gera MJCF do campo (14x9m, bola Size-5, gols) a partir do robô
play_mujoco_soccer_walk.py      # viewer interativo (MJ.5)
scripts/render_walk_on_field.py # render headless -> MP4
configs/Base_Walk_Extended.yaml # config da política de walk (do htwk-gym)
models/base_walk_extended.pt    # política JIT pré-treinada (do htwk-gym)
resources/T1/                    # MJCF + meshes do T1
tests/unit/                      # testes de cena, física da bola e política no campo
docs/                            # requisitos e roadmap
```

## Nota de simulador
Treino usa **Isaac Gym** (Linux + GPU NVIDIA). MuJoCo é usado para **validação
cross-sim** e roda nativo no Mac. Ver `docs/01-parte1-mvp.md` (Requisitos de hardware).
