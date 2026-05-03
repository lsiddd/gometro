# Evolucao de Pesos de Recompensa

Este guia descreve como usar o pipeline evolutivo para otimizar os pesos de recompensa do agente RL.

O objetivo e encontrar um `RewardConfig` que treine melhor o PPO, mas selecionar os candidatos por metricas externas do jogo:

- semanas sobrevividas;
- score;
- passageiros entregues;
- estabilidade entre episodios;
- baixa taxa de acoes invalidas.

O reward interno continua sendo usado pelo PPO como sinal de aprendizado. A selecao evolutiva, porem, usa a fitness calculada por `eval_suite.py`, nao o `episode_reward`.

## Visao Geral

O fluxo completo e:

```text
1. Gerar populacao de RewardConfig
2. Treinar um PPO curto para cada individuo
3. Avaliar cada checkpoint por score/weeks/passengers
4. Selecionar elites
5. Gerar nova populacao por copia, mutacao, crossover e individuos aleatorios
6. Repetir por varias geracoes
7. Re-treinar os melhores configs com budget longo
8. Treinar o modelo final com o melhor RewardConfig
9. Avaliar e exportar para ONNX
```

Arquivos principais:

- `python/evolve_rewards.py`: orquestra populacao, treino, avaliacao e re-treino.
- `python/eval_suite.py`: avalia checkpoints usando fitness externa.
- `python/train.py`: treina PPO com `--reward-config`.
- `docs/evolving_reward_weights_plan.md`: plano tecnico de implementacao.

## O Que Evolui

Cada individuo e um JSON com os pesos:

```json
{
  "per_passenger": 3.0,
  "queue_coeff": 0.03,
  "queue_delta_coeff": 0.2,
  "overcrowd_coeff": 0.75,
  "overcrowd_delta_coeff": 2.0,
  "danger_thresh": 0.8,
  "danger_penalty": 0.5,
  "noop_critical_penalty": 0.25,
  "week_bonus": 20.0,
  "terminal_penalty": 100.0,
  "invalid_action": 1.0
}
```

Esses valores sao aplicados no servidor Go via gRPC antes dos resets de treino e avaliacao.

## Fitness

A fitness padrao atual (`--fitness-version v2`) e:

```text
fitness =
  1000.0 * mean_week
+    2.0 * mean_score
+    1.0 * mean_passengers_delivered
-  250.0 * std_week
-  100.0 * mean_invalid_action_rate
-  400.0 * mean_noop_rate
-   40.0 * mean_queue_pressure
-  120.0 * mean_overcrowd_pressure
-  100.0 * mean_danger_count
-  500.0 * zero_throughput_rate
```

Isso prioriza sobreviver mais semanas, mas penaliza politicas passivas que apenas esperam o jogo terminar. `zero_throughput_rate` mede a fracao de episodios com score ou passageiros entregues igual a zero.

Os pesos podem ser alterados:

```bash
--week-weight 1000
--score-weight 2
--passenger-weight 1
--std-week-penalty 250
--invalid-action-penalty 100
--noop-penalty 400
--queue-penalty 40
--overcrowd-penalty 120
--danger-penalty 100
--zero-throughput-penalty 500
```

Use `--fitness-version v1` apenas para comparar com runs antigas.

## Preparacao

Na raiz do repositorio:

```bash
just build
```

Depois entre na pasta Python:

```bash
cd python
```

## Smoke Test

Antes de rodar uma busca cara, valide o pipeline com budget pequeno:

```bash
just evolve-smoke
```

Equivalente manual:

```bash
uv run python evolve_rewards.py \
  --run-dir evolution_runs/reward_search_smoke \
  --population 3 \
  --generations 1 \
  --elites 1 \
  --random-individuals 1 \
  --train-timesteps 10000 \
  --learn-chunk 10000 \
  --eval-episodes 2 \
  --eval-max-steps 500 \
  --n-envs 2 \
  --retrain-top 1 \
  --retrain-timesteps 10000
```

Se o smoke passar, confira:

```bash
cat evolution_runs/reward_search_smoke/final_summary.json
```

## Busca Evolutiva Inicial

Comando recomendado para uma primeira busca real:

```bash
just evolve
```

Para sobrescrever parametros principais:

```bash
just evolve evolution_runs/minha_busca 16 12 1000000 64
```

Equivalente manual:

```bash
uv run python evolve_rewards.py \
  --run-dir evolution_runs/reward_search_001 \
  --population 12 \
  --generations 10 \
  --elites 3 \
  --random-individuals 2 \
  --sigma 0.25 \
  --train-timesteps 500000 \
  --learn-chunk 500000 \
  --n-envs 8 \
  --eval-episodes 32 \
  --eval-max-steps 4000 \
  --eval-cities london \
  --eval-complexities 4 \
  --eval-spawn-factors 1.0 \
  --eval-seeds 101,202,303,404 \
  --retrain-top 3 \
  --retrain-timesteps 20000000 \
  --retrain-eval-episodes 64 \
  --retrain-eval-cities london,paris,newyork \
  --retrain-eval-spawn-factors 1.25,1.0
```

Saidas importantes:

```text
evolution_runs/reward_search_001/
  run_config.json
  generation_000/
    individual_000/
      reward_config.json
      manifest.json
      train.log
      eval.log
      eval.json
      checkpoints/
      tb_logs/
    summary.json
  baseline_default/
    summary.json
  events.jsonl
  individuals.csv
  generations.csv
  leaderboard.json
  dashboard.html
  retrain/
    finalist_000/
      reward_config.json
      manifest.json
      result.json
      checkpoints/
      tb_logs/
    summary.json
  final_summary.json
```

Ver melhores resultados:

```bash
cat evolution_runs/reward_search_001/final_summary.json
cat evolution_runs/reward_search_001/retrain/summary.json
```

O melhor `RewardConfig` re-treinado normalmente estara em:

```text
evolution_runs/reward_search_001/retrain/finalist_000/reward_config.json
```

## Guardrails

O script registra avisos em `run_config.json` quando a avaliacao parece fraca demais.

Exemplos:

- poucos episodios de avaliacao;
- re-treino validando no mesmo benchmark da evolucao;
- poucas cidades na avaliacao.

Durante a busca, candidatos tambem podem ser marcados como `pruned` e excluidos da elite quando:

- `mean_noop_rate` passa de `--prune-noop-rate`;
- `zero_throughput_rate` passa de `--prune-zero-throughput-rate`;
- a fitness fica muito abaixo do baseline default pelo limite `--prune-default-delta`.

O baseline default e treinado e avaliado no mesmo benchmark por padrao. Use `--skip-baseline` apenas para smoke tests rapidos.

## Observabilidade Local

Cada run grava:

- `events.jsonl`: eventos estruturados de run, geracao, individuo, pruning e baseline;
- `individuals.csv`: tabela plana de todos os individuos avaliados;
- `generations.csv`: melhor/media de fitness e contagem de pruning por geracao;
- `leaderboard.json`: top individuos globais atualizado ao longo da run;
- `dashboard.html`: relatorio local simples para abrir no navegador.

Para transformar esses avisos em erro:

```bash
--strict-guardrails
```

Exemplo:

```bash
uv run python evolve_rewards.py \
  --run-dir evolution_runs/reward_search_strict \
  --population 12 \
  --generations 10 \
  --eval-episodes 32 \
  --strict-guardrails
```

## Treino Final Com o Melhor Config

Depois de escolher o melhor `reward_config.json`, rode um treino longo:

```bash
uv run python train.py \
  --n-envs 12 \
  --city london \
  --reward-config evolution_runs/reward_search_001/retrain/finalist_000/reward_config.json \
  --total-timesteps 100000000 \
  --learn-chunk 500000 \
  --checkpoint-dir checkpoints/final_reward_optimized \
  --log-dir tb_logs/final_reward_optimized \
  --base-port 8765
```

O checkpoint final sera:

```text
checkpoints/final_reward_optimized/minimetro_latest.zip
```

## Avaliacao Final

Avalie em mais cidades e mais episodios:

```bash
uv run python eval_suite.py \
  --model checkpoints/final_reward_optimized/minimetro_latest.zip \
  --output checkpoints/final_reward_optimized/eval.json \
  --episodes 128 \
  --max-steps 5000 \
  --cities london,paris,newyork,tokyo \
  --complexities 4 \
  --spawn-factors 1.25,1.0
```

Ver resumo:

```bash
cat checkpoints/final_reward_optimized/eval.json
```

## Exportar Para ONNX

```bash
uv run python export_onnx.py \
  --model checkpoints/final_reward_optimized/minimetro_latest.zip \
  --output actor_reward_optimized.onnx
```

## Rodar no Jogo

Terminal 1:

```bash
cd python
uv run python infer.py --model actor_reward_optimized.onnx --port 9000
```

Terminal 2, na raiz do repo:

```bash
./minimetro --rl-client localhost:9000
```

## Dicas Praticas

- Comece com smoke pequeno. Nao rode busca longa antes de validar paths, ports e artefatos.
- Compare sempre contra o reward default.
- Use `retrain-top` para evitar escolher configs que so aprendem rapido no budget curto.
- Use avaliacao final em cidades que nao foram usadas na selecao principal.
- Se a busca estiver instavel, aumente `--eval-episodes`.
- Se a populacao convergir cedo, aumente `--random-individuals` ou `--sigma`.
- Se as mutacoes forem agressivas demais, reduza `--sigma`.

## Comando Curto Para Comparar um Config Manual

Treinar com um config especifico:

```bash
uv run python train.py \
  --n-envs 8 \
  --city london \
  --reward-config path/to/reward_config.json \
  --total-timesteps 5000000 \
  --checkpoint-dir checkpoints/manual_reward_test \
  --log-dir tb_logs/manual_reward_test
```

Avaliar:

```bash
uv run python eval_suite.py \
  --model checkpoints/manual_reward_test/minimetro_latest.zip \
  --output checkpoints/manual_reward_test/eval.json \
  --episodes 64 \
  --cities london,paris,newyork \
  --complexities 4 \
  --spawn-factors 1.0
```
