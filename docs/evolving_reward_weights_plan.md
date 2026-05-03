# Plano: Evolucao dos Pesos de Recompensa para RL

## Objetivo

Implementar um pipeline evolutivo para otimizar os pesos de recompensa e penalidades usados no treinamento PPO do agente Mini Metro. A selecao dos individuos nao deve usar o `episode_reward` interno, e sim uma fitness externa baseada em desempenho real:

- `weeks_survived`
- `score`
- `passengers_delivered`
- robustez entre seeds/cidades/dificuldades
- baixa taxa de acoes invalidas

A ideia central e separar duas coisas:

- **Reward shaping**: sinal denso usado pelo PPO durante o treinamento.
- **Fitness evolutivo**: criterio externo que mede se aquele reward shaping produz um agente melhor no jogo real.

## Arquitetura Alvo

O sistema final tera quatro componentes:

1. `RewardConfig` configuravel no ambiente Go.
2. Endpoint gRPC de controle para aplicar pesos de recompensa por experimento.
3. Suite de avaliacao Python com metricas externas robustas.
4. Script evolutivo Python que treina populacoes de PPO com diferentes `RewardConfig`.

Fluxo:

```text
evolve_rewards.py
  -> amostra RewardConfig
  -> inicia treino PPO com esse config
  -> aplica RewardConfig via gRPC
  -> treina por budget fixo
  -> avalia por score/weeks em benchmark externo
  -> seleciona/muta/crossover
  -> repete
```

## Fase 1: Tornar RewardConfig Dinamico no Go

Status: concluida.

Implementado em `rl/env.go`: `RewardConfig`, `DefaultRewardConfig`, `ValidateRewardConfig` e `SetRewardConfig`. Os calculos de reward agora usam `e.rewardCfg`, e os testes em `rl/env_test.go` cobrem alteracao dos coeficientes e rejeicao de valores invalidos.

Hoje os pesos vivem como `const` em `rl/env.go`. Isso obriga recompilar para cada individuo, o que inviabiliza evolucao eficiente. O primeiro passo e transformar esses valores em configuracao por ambiente.

Adicionar em `rl/env.go`:

```go
type RewardConfig struct {
    PerPassenger        float64
    QueueCoeff          float64
    QueueDeltaCoeff     float64
    OvercrowdCoeff      float64
    OvercrowdDeltaCoeff float64
    DangerThresh        float64
    DangerPenalty       float64
    NoOpCriticalPenalty float64
    WeekBonus           float64
    TerminalPenalty     float64
    InvalidAction       float64
}
```

Adicionar:

```go
func DefaultRewardConfig() RewardConfig
func (e *RLEnv) SetRewardConfig(cfg RewardConfig)
```

Substituir o uso dos `const` atuais por `e.rewardCfg.*` dentro de:

- `computeReward`
- `rewardStatePressure`
- `computeNoOpPenalty`
- penalidade de acao invalida em `Step`

Manter limites defensivos:

- coeficientes negativos devem ser rejeitados ou clampados para zero;
- `DangerThresh` deve ficar em `[0.1, 1.5]`;
- `TerminalPenalty` deve ter limite superior razoavel para nao destruir PPO;
- valores NaN/Inf devem ser rejeitados.

## Fase 2: Controle gRPC para RewardConfig

Status: concluida.

Implementado com `service Control` tipado em `rl/proto/minimetro.proto`, stubs Go/Python regenerados, `pb.RegisterControlServer` no `cmd/rl_server`, e `SetRewardConfig` em `rl/grpc_service.go`. O config atual agora e propagado para o env single, envs vetorizados existentes, novos `ResetVector` e auto-reset nativo.

Adicionar um endpoint de controle similar a `SetDifficulty` e `SetComplexity`.

Opcoes:

1. Caminho rapido: registrar manualmente um metodo `SetRewardConfig` no servico `rl.Control`, como ja e feito em `rl/grpc_service.go`.
2. Caminho limpo: alterar `rl/proto/minimetro.proto` e regenerar stubs.

Recomendacao: usar o caminho limpo, porque `RewardConfig` tem muitos campos e merece contrato tipado.

Adicionar mensagem no protobuf:

```proto
message RewardConfigRequest {
  double per_passenger = 1;
  double queue_coeff = 2;
  double queue_delta_coeff = 3;
  double overcrowd_coeff = 4;
  double overcrowd_delta_coeff = 5;
  double danger_thresh = 6;
  double danger_penalty = 7;
  double noop_critical_penalty = 8;
  double week_bonus = 9;
  double terminal_penalty = 10;
  double invalid_action = 11;
}
```

Propagar o config para:

- `s.env`
- todos os `s.vecEnvs`
- novos envs criados em `ResetVector`
- envs resetados automaticamente em `RunVectorEpisode`

Testes Go:

- `TestSetRewardConfigAppliesToResetVector`
- `TestRewardConfigChangesRewardBreakdown`
- `TestRewardConfigRejectsInvalidValues`

## Fase 3: API Python no VecEnv

Status: concluida.

Implementado em `python/env.py`: `reward_config_request`, `MiniMetroVecEnv.set_reward_config`, `env_method("set_reward_config", ...)`, `get_attr`/`set_attr` e uso do `ControlStub` gerado. Implementado em `python/train.py`: `load_reward_config` e argumento `--reward-config`, aplicado aos ambientes de treino e avaliacao antes do primeiro reset/wrapper.

Adicionar em `python/env.py`:

```python
def env_method("set_reward_config", cfg: dict)
```

Ou metodo direto:

```python
def set_reward_config(self, cfg: RewardConfigDict) -> None
```

O script de treino deve aceitar um caminho JSON:

```bash
uv run python train.py --reward-config reward_configs/individual_003.json
```

Ao iniciar o ambiente:

1. conectar ao server;
2. validar constantes;
3. aplicar `set_reward_config`;
4. entao chamar `reset`.

Isso evita que o primeiro rollout seja feito com o reward default.

## Fase 4: Suite de Avaliacao Externa

Status: concluida.

Implementado em `python/eval_suite.py`: avaliacao por matriz de cidade, complexidade e `spawn_rate_factor`, metricas por episodio, resumo agregado, fitness externa e saida JSON opcional. Testes em `python/test_eval_suite.py` cobrem resumo, fitness e parsing de listas.

Criar `python/eval_suite.py`.

Ela deve carregar um checkpoint e avaliar sem usar reward interno como criterio principal.

Metricas por episodio:

- `score`
- `week`
- `passengers_delivered`
- `steps`
- `stations`
- `invalid_action_rate`
- `noop_rate`
- `danger_count_mean`
- `queue_pressure_mean`
- `overcrowd_pressure_mean`

Config benchmark inicial:

```text
cities: london
complexity: 4
spawn_rate_factor: 1.0
episodes_per_candidate: 32
max_steps_per_episode: 4000
deterministic: true
```

Depois expandir:

```text
cities: london + outras cidades ja suportadas
spawn_rate_factor: [1.25, 1.0]
complexity: [3, 4]
seeds: quando o ambiente expuser seed explicita
```

Fitness inicial:

```text
fitness =
  1000.0 * mean_week
+    1.0 * mean_score
+    0.5 * mean_passengers_delivered
-  200.0 * std_week
-   50.0 * invalid_action_rate
```

Motivo: `week` mede sobrevivencia, `score` mede eficiencia, `std_week` penaliza estrategias instaveis.

## Fase 5: Evolucao dos Reward Weights

Status: concluida.

Implementado em `python/evolve_rewards.py`: populacao inicial, mutacao, crossover, elitismo, individuos aleatorios, execucao sequencial de `train.py`/`eval_suite.py`, artefatos por individuo e `--dry-run` para smoke tests. `python/train.py` tambem recebeu `--total-timesteps`, `--learn-chunk`, `--checkpoint-dir`, `--log-dir` e `--base-port` para permitir budgets finitos e isolamento por individuo.

Criar `python/evolve_rewards.py`.

Comecar simples, sem framework externo:

- populacao: 12 individuos
- elites: top 3 preservados
- filhos: mutacao dos elites
- aleatorios: 2 por geracao para exploracao
- budget por individuo: 500k a 2M timesteps
- geracoes iniciais: 10

Depois trocar para CMA-ES ou Nevergrad se necessario.

### Genotipo

Representar coeficientes positivos em espaco log:

```python
gene = log(value)
value = exp(gene)
```

Isso evita que mutacoes tenham escala ruim.

Faixas iniciais:

```text
PerPassenger:        0.5 .. 10
QueueCoeff:          0.001 .. 0.2
QueueDeltaCoeff:     0.01 .. 2
OvercrowdCoeff:      0.05 .. 5
OvercrowdDeltaCoeff: 0.05 .. 10
DangerThresh:        0.5 .. 1.1
DangerPenalty:       0.05 .. 5
NoOpCriticalPenalty: 0.0 .. 2
WeekBonus:           1 .. 100
TerminalPenalty:     10 .. 500
InvalidAction:       0.1 .. 10
```

### Mutacao

Para coeficientes log:

```python
child_gene = parent_gene + normal(0, sigma)
```

Com `sigma` inicial entre `0.15` e `0.35`.

Para `DangerThresh`, usar mutacao direta com clamp.

### Crossover

Uniform crossover entre dois elites:

```python
child[i] = parent_a[i] if rand() < 0.5 else parent_b[i]
```

Depois aplicar mutacao leve.

## Fase 6: Isolamento de Experimentos

Status: concluida.

Implementado em `python/evolve_rewards.py`: `run_config.json` com argumentos, comando, executavel Python e metadados Git; `manifest.json` por individuo com paths, comandos de treino/avaliacao e dados completos do individuo; `origin`, `parents`, `metadata`, `mutation_sigma` e `mutation_delta` para auditar linhagem e mutacoes.

Cada individuo deve ter diretorios proprios:

```text
python/evolution_runs/<run_id>/
  generation_000/
    individual_000/
      reward_config.json
      checkpoints/
      tb_logs/
      eval.json
      train.log
```

Registrar tambem:

- parent ids
- mutacoes aplicadas
- seed Python
- commit Git
- comando usado
- metricas completas de avaliacao

Isso torna possivel retomar, comparar e auditar resultados.

## Fase 7: Re-treino dos Melhores

Status: concluida.

Implementado em `python/evolve_rewards.py`: `--retrain-top`, comandos separados para re-treino e avaliacao dos finalistas, diretorio `retrain/finalist_NNN`, manifestos, `result.json`, `retrain/summary.json` e inclusao de `best_retrained` no `final_summary.json`. O re-treino respeita `--dry-run` para validar a orquestracao sem custo de treino.

O treino curto por individuo serve para estimar potencial, mas pode favorecer configs que aprendem rapido e nao configs que convergem melhor.

Depois de 10 a 20 geracoes:

1. pegar top 5 `RewardConfig`;
2. treinar cada um do zero com budget longo, por exemplo 20M a 100M timesteps;
3. avaliar com suite maior;
4. selecionar checkpoint final por fitness externa.

Opcional:

- fazer ensemble de configs para gerar rollouts;
- destilar comportamento em uma unica policy;
- fine-tunar a melhor policy com reward default ou config vencedora.

## Fase 8: Robustez e Anti-Overfitting

Status: concluida.

Implementado em `python/eval_suite.py`: pesos de fitness configuraveis por CLI para sobrevivencia, score, passageiros, penalidade de variancia e penalidade de acoes invalidas. Implementado em `python/evolve_rewards.py`: guardrails para numero minimo de episodios/cidades, `--strict-guardrails`, avisos quando re-treino usa o mesmo benchmark da evolucao e registro desses avisos no `run_config.json`.

Riscos:

- reward config overfita em Londres;
- reward config maximiza score curto e morre cedo;
- reward config explora shaping e nao jogo real;
- configs com penalidades enormes instabilizam PPO;
- comparacoes injustas por variancia de seeds.

Mitigacoes:

- fitness externa nunca usa `episode_reward`;
- avaliar em multiplos episodios;
- penalizar variancia;
- manter conjunto de validacao separado;
- retreinar finalistas do zero;
- usar evaluation deterministic e, depois, stochastic tambem;
- manter limites conservadores para coeficientes.

## Ordem de Implementacao

1. Criar `RewardConfig` no Go e substituir constantes.
2. Adicionar testes unitarios para reward config.
3. Expor `SetRewardConfig` via protobuf/gRPC.
4. Adicionar suporte Python em `MiniMetroVecEnv`.
5. Adicionar `--reward-config` em `train.py`.
6. Criar `eval_suite.py` com JSON de saida.
7. Criar `evolve_rewards.py` com algoritmo genetico simples.
8. Rodar smoke test com populacao 2, geracao 1, budget pequeno.
9. Rodar primeira busca real com populacao 12, 10 geracoes.
10. Retreinar finalistas com budget longo.

## Criterios de Aceite

Implementacao minima esta pronta quando:

- `go test ./...` passa.
- `uv run pytest` passa em `python/`.
- `train.py --reward-config ...` aplica pesos sem recompilar Go.
- `eval_suite.py` gera `eval.json` com fitness externa.
- `evolve_rewards.py` consegue completar uma geracao pequena.
- Cada individuo salva config, checkpoint e metricas.

Sucesso experimental inicial:

- pelo menos um `RewardConfig` evoluido supera o baseline default em `mean_week`;
- a melhora permanece apos retreino do zero;
- a melhora aparece em avaliacao externa sem usar reward interno.

## Extensoes Futuras

- Trocar GA simples por CMA-ES ou Nevergrad.
- Evoluir tambem hiperparametros PPO junto com reward weights.
- Usar MAP-Elites com descritores de comportamento.
- Adicionar seeds proceduralmente controlaveis no Go.
- Usar Prioritized Level Replay quando houver seeds/levels.
- Fazer Population-Based Training onde agentes herdam pesos PPO e `RewardConfig`.
