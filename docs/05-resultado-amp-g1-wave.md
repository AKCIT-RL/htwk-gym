# Resultado do gate AMP G1 wave

Data: 2026-07-20

## Veredito

O pipeline de dados, Docker, Isaac Gym, GPU PhysX, viewer, checkpoint e logging está operacional. O overfit AMP da motion única ainda **não passou**; multimotion permanece bloqueado.

## Contratos aprovados

- Asset G1: 38 corpos, 29 DOFs, limites finitos e não degenerados.
- Motion: `3630 x 35`, `float32`, 239 FPS, 15.1841 s, `CLAMP`.
- Atividade de braços: 77.82% dos frames acima de 0.25 rad/s; velocidade média conjunta 4.12 rad/s.
- Runtime: Python 3.8, Isaac Gym Preview 4, PyTorch 2.4.1+cu118 e RTX 3060.
- Smoke AMP: checkpoint recarregável, logs e perdas finitas.

## Correções implementadas

1. Resets opcionais truncados por horizonte para evitar a cauda `CLAMP`.
2. Referência cinemática atualizada também em headless para diagnósticos válidos.
3. Retorno de teste AMP calculado pelo discriminador.
4. Normalizador de demonstrações deixa de acumular após o limite configurado.
5. Fail-fast para ganhos PD e torque em ambientes DeepMimic/AMP.
6. Override explícito de PD no environment G1 wave, porque o importador MJCF do Isaac Gym ignora `stiffness`/`damping` do XML.
7. Esforços dos joints são preservados como `gear` dos motores durante a sanitização do MJCF.

## Experimentos executados

| Experimento | Amostras | Test episode length inicial/final | Disc reward final | Disc acc agente/demo final | Resultado |
|---|---:|---:|---:|---:|---|
| Sem PD | 5,005,312 | 21.41 / 21.60 | 0.281 | 0.959 / 1.000 | Reprovado |
| PD 400/40, reset truncado | 253,952 | 35.11 / 37.29 | 0.413 | 0.957 / 1.000 | Reprovado |
| PD 400/40, reset completo | 253,952 | ~35 / 37.42 | 0.480 | 0.929 / 1.000 | Reprovado |

O PD melhorou a sobrevivência em aproximadamente 74%, provando que a versão anterior não possuía autoridade física adequada. Mesmo assim, o gate de promoção exigia mediana >=45 passos, melhor >=60 e discriminador não saturado; nenhum piloto atingiu esses critérios.

## Bloqueios para multimotion

1. O single-motion AMP ainda não demonstrou overfit.
2. Existe somente uma motion G1/MimicKit convertida e validada.
3. Os outros 21 BVHs são apenas fontes humanas; precisam de retarget, validação e manifesto antes de treino.

## Próximo gate

Investigar estabilidade passiva e tracking PD fora do AMP: executar rollout determinístico que mantém os alvos da própria motion e medir sobrevivência/erro. Se o controlador de referência não sustentar o G1 por pelo menos 90% do horizonte, ajustar somente o perfil PD por grupos de joints antes de novo experimento AMP.
