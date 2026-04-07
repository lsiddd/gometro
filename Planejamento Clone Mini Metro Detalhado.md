# **Arquitetura e Especificações Técnicas para a Reconstrução Sistêmica de Mini Metro**

O desenvolvimento de uma simulação de trânsito minimalista, baseada nos preceitos estabelecidos por Mini Metro, exige uma compreensão profunda da convergência entre o design visual abstrato e a engenharia de sistemas complexos. A premissa central do jogo não reside meramente na construção de linhas ferroviárias, mas na otimização de restrições sob condições de incerteza e crescimento procedural.1 Para o planejamento de um clone exato, é imperativo decompor a obra em seus pilares fundamentais: a filosofia de design topológico, a mecânica de malha de grafos dinâmicos, o motor de geração procedural de demanda e a arquitetura de áudio reativo. Este relatório detalha cada componente necessário para replicar a experiência com fidelidade técnica e comportamental.

## **Filosofia de Design e a Transição do Topográfico para o Topológico**

A gênese de Mini Metro está profundamente enraizada na obra de Harry Beck, o desenhista técnico que revolucionou o mapa do metrô de Londres em 1933\.2 Ao contrário dos mapas anteriores que tentavam manter a precisão geográfica, o modelo de Beck priorizava a topologia — a relação entre os nós e a sequência de paradas — em detrimento da distância real.4 Para um desenvolvedor que planeja um clone, entender essa distinção é o primeiro passo para a implementação da interface de usuário (UI) e da experiência do usuário (UX).  
A lógica de Beck baseava-se em diagramas de circuitos elétricos, eliminando a "clutter" geográfica que tornava os mapas ilegíveis à medida que a rede se expandia.4 No jogo, essa filosofia é traduzida em restrições visuais rígidas: as linhas de metrô só podem ser desenhadas em ângulos de 0°, 45° e 90°.6 Essa restrição não é meramente estética; ela simplifica o modelo mental do jogador e permite que o sistema de renderização utilize splines lineares e curvas suaves de forma previsível.8  
A cidade em Mini Metro é uma entidade que cresce de forma desordenada, forçando o sistema de transporte a ser puramente reativo.10 Isso cria um ambiente de planejamento com informações incompletas, onde o jogador deve construir sistemas flexíveis o suficiente para se adaptar a requisitos futuros desconhecidos.1 O design ensina a gestão de dívida técnica: soluções rápidas e ineficientes no início do jogo tornam-se gargalos catastróficos à medida que a densidade de passageiros aumenta, forçando a iteração audaciosa — muitas vezes exigindo a destruição total de uma linha para reconstruí-la com uma logística superior.1

## **Estrutura do Loop de Jogo e Ciclo de Vida Sistêmico**

A arquitetura do loop de jogo segue um modelo clássico de desafio-ação-recompensa, operando em múltiplas escalas temporais.11 O micro-loop ocorre a cada poucos segundos, quando novos passageiros surgem ou trens chegam às estações. O meta-loop é definido pelo ciclo semanal, onde o jogador recebe recursos adicionais para expandir a rede.12

### **O Ciclo Semanal e Gestão de Recursos**

O progresso é marcado pela passagem do tempo in-game, culminando em um "aumento de orçamento" a cada domingo.12 O sistema apresenta ao jogador uma escolha estratégica entre dois ativos, forçando uma decisão baseada na análise preditiva das necessidades da rede.12

| Recurso | Função Primária | Impacto Estratégico |
| :---- | :---- | :---- |
| Locomotiva | Unidade básica de tração. | Essencial para abrir novas linhas ou aumentar a frequência em rotas saturadas. 13 |
| Vagão | Extensão de capacidade para locomotivas. | Aumenta o rendimento de uma linha sem ocupar um slot de trem adicional. 13 |
| Nova Linha | Desbloqueia uma nova cor e frota. | Permite a criação de rotas alternativas e reduz a dependência de transferências complexas. 13 |
| Túnel / Ponte | Permite cruzamento de corpos d'água. | Um limitador geográfico crítico que define a viabilidade de expansão para certas ilhas. 13 |
| Interchange | Upgrade de estação de alta capacidade. | Aumenta o limite de superpopulação e a velocidade de transferência de passageiros. 14 |

Este sistema de recompensas cria uma tensão constante. Escolher uma nova linha oferece flexibilidade, mas sem vagões suficientes, as linhas existentes podem entrar em colapso. O planejamento de um clone deve garantir que essas escolhas não sejam triviais, mas sim respostas diretas aos gargalos identificados pelo jogador durante a semana anterior.1

## **Mecânicas de Estações e Passageiros: O Sistema de Formas**

A iconografia de Mini Metro utiliza formas geométricas para representar a demanda e a oferta de transporte. Esta abstração esconde uma simulação de zonas urbanas.10

### **Tipologia de Estações e Raridade**

As estações aparecem proceduralmente nas bordas do mapa conforme a cidade se expande.13 Elas são classificadas por sua frequência e papel no ecossistema de transporte.

| Forma | Categoria | Interpretação Sugerida | Comportamento |
| :---- | :---- | :---- | :---- |
| Círculo | Comum | Áreas Residenciais | Fonte primária de passageiros; o tipo mais abundante. 10 |
| Triângulo | Comum | Atividade Comercial | Frequentemente o destino de passageiros vindos de círculos. 10 |
| Quadrado | Raro | Centros de Negócios | Menos comuns; atuam como hubs naturais para a rede. 17 |
| Únicas (Estrela, Cruz, etc.) | Únicas | Instituições Especializadas | Apenas uma de cada tipo por mapa; criam pontos de pressão logística extrema. 10 |

A relevância dessas formas para a reconstrução do jogo reside na lógica de "matching". Um passageiro em forma de triângulo só pode desembarcar em uma estação triangular.13 Se o trem passar por estações circulares, o passageiro permanecerá a bordo, ocupando espaço valioso. Isso introduz o conceito de alternância de estações: uma linha eficiente deve, idealmente, conectar formas diferentes em sequência (ex: Círculo-Triângulo-Círculo-Quadrado) para maximizar o fluxo de passageiros entrando e saindo do sistema em cada parada.16

### **Comportamento e IA de Passageiros**

A inteligência artificial dos passageiros é baseada em algoritmos de busca de caminho (pathfinding), como A\* ou BFS (Breadth-First Search).7 Ao surgir em uma estação, o passageiro avalia a rede disponível para encontrar a rota mais rápida até seu destino.  
Fatores decisivos na escolha da rota incluem o tempo estimado de viagem, o tempo de espera por conexões e um custo de transferência inflado.23 O sistema penaliza transferências excessivas, fazendo com que os passageiros prefiram rotas diretas, mesmo que ligeiramente mais longas no papel.22 Além disso, os passageiros são capazes de reservar assentos em trens que ainda não chegaram, evitando que vinte pessoas tentem entrar em um vagão que só comporta seis.23  
Uma característica crítica para o clone é a reavaliação dinâmica de rota. Se a estação onde o passageiro se encontra começa a superlotar, ele pode optar por um caminho sub-ótimo apenas para aliviar a pressão no nó atual.23 Esse comportamento ajuda a evitar falhas catastróficas imediatas, dando ao jogador uma janela de tempo para reagir.

## **Motor de Geração Procedural e Dinâmica Espacial**

Para que um clone seja exato, ele deve replicar a maneira como o mundo de Mini Metro é construído e como ele evolui. A geração procedural governa tanto a geografia quanto o cronograma de eventos.6

### **Geração de Terreno e Mapas**

Os mapas são baseados em cidades reais, mas os detalhes geográficos são aleatorizados usando sementes (seeds) de RNG.6

* **Rios e Costas:** São frequentemente gerados usando algoritmos como "Drunkard's Walk" para rios sinuosos ou ruído de Perlin para arquipélagos.6 A água atua como um bloqueador físico que exige o consumo de túneis para a travessia de linhas.13  
* **Camadas de Densidade:** O motor divide o mapa em uma grade onde cada célula possui valores de densidade residencial e comercial. Isso influencia onde as estações têm maior probabilidade de surgir e que tipo de passageiros elas gerarão.6

### **Agendamento de Estações (Scheduling)**

O surgimento de estações não é puramente aleatório. O sistema utiliza um pool fixo de formas para garantir que a rede permaneça funcional.25 Se houvesse apenas estações circulares, o jogo terminaria instantaneamente por falta de destinos. O agendamento segue padrões que podem ser controlados por parâmetros específicos da cidade, como velocidade de crescimento e afinidade de zona.26  
A probabilidade de uma nova estação ser de um determinado tipo pode ser modelada matematicamente para evitar o agrupamento excessivo de formas idênticas. Por exemplo, se $C$, $S$ e $T$ representam o número atual de círculos, quadrados e triângulos, a probabilidade de o próximo nó ser um círculo $P(C)$ diminui conforme $C$ aumenta 25:

$$P(C) \= \\frac{S \+ T}{2(C \+ S \+ T)}$$  
Este equilíbrio dinâmico garante que, embora o jogador se sinta pressionado pela aleatoriedade, o sistema está secretamente fornecendo as peças necessárias para a sobrevivência, desde que a logística seja bem planejada.

## **Implementação Técnica: Gráficos, Malhas e Interação**

A estética minimalista de Mini Metro esconde uma complexidade técnica na renderização de linhas e na interação com o usuário. A utilização de motores modernos como Unity exige uma abordagem específica para a geração de malhas (meshes) procedurais.9

### **Renderização de Linhas e Splines**

As linhas de metrô não são apenas vetores 2D; elas são malhas 3D extrudadas ao longo de splines.9 Para replicar esse comportamento em um clone:

1. **Amostragem da Spline:** A curva entre duas estações é dividida em pequenos segmentos. Quanto mais amostras, mais suave será a curva.9  
2. **Geração de Vértices:** Para cada ponto na spline, o sistema deve calcular dois vértices perpendiculares à direção da linha, afastados por uma distância igual à metade da largura da linha ($W/2$).30  
3. **Mapeamento de Triângulos:** Os vértices são conectados em grupos de três para formar a face da malha. O uso do sentido horário na definição dos triângulos garante que a face frontal seja renderizada corretamente pela GPU.31  
4. **Snapping e Ângulos:** O sistema de interação deve forçar o desenho das linhas em incrementos de 45°.6 Se duas estações não estiverem perfeitamente alinhadas, o código de desenho deve gerar automaticamente uma conexão composta por um segmento ortogonal e um diagonal.32

### **Interação e Manipulação de Linhas**

A fluidez da interface é um dos maiores diferenciais do jogo original. O jogador pode clicar no meio de uma linha e arrastá-la para uma nova estação sem precisar redesenhar todo o percurso.33 Tecnicamente, isso envolve detectar colisões com "trigger colliders" nas estações e atualizar dinamicamente a lista de nós que compõem o grafo daquela linha.8 Se uma linha passa por cima de uma estação mas não está conectada a ela, o visual deve refletir isso através de uma pequena divisão central na malha da linha, indicando que o trem não parará ali.35

## **Arquitetura de Áudio Procedural e Reativo**

O áudio em Mini Metro, criado por Disasterpeace, é uma extensão direta dos dados do jogo, utilizando princípios de serialismo musical.36 Para um clone exato, o som não deve ser apenas uma trilha de fundo, mas uma ferramenta de feedback diagnóstico.

### **Sonificação de Dados**

Cada ação no jogo gera um evento musical. O sistema mapeia variáveis de jogo para eixos sonoros:

* **Ritmo:** A pulsação de cada linha é determinada pelo número de estações que ela atende. Linhas mais longas têm sequências rítmicas mais complexas.36  
* **Timbre:** O tipo de estação (Círculo, Triângulo, etc.) determina a qualidade tonal do som produzido quando um passageiro embarca ou desembarca.36  
* **Dinâmica e Pan:** A posição visual da estação na tela controla o balanço estéreo (panning), enquanto a taxa de ocupação da estação controla a intensidade do som.36  
* **Velocidade do Relógio:** A trilha sonora deve se ajustar às quatro velocidades do jogo (Pausa, Normal, Rápido e Pausa de Rápido). Em velocidades maiores, os ataques dos sons devem ser mais curtos para evitar cacofonia.36

O uso de plugins como o G-Audio para Unity permite que o loop de atualização do jogo funcione como um sequenciador musical, disparando amostras de ondas senoidais puras com variações aleatórias controladas para manter a sonoridade orgânica e minimalista.37

## **Falha Sistêmica: A Lógica da Superpopulação**

O estado de derrota é o mecanismo que encerra a experiência e define a pontuação do jogador. Ele é ativado quando a demanda supera a capacidade de processamento de passageiros da rede.13

### **O Temporizador de Overcrowding**

Cada estação comum tem uma capacidade base de 6 passageiros.13 Quando o 7º passageiro chega, um temporizador visual (um círculo cinza que se preenche) é iniciado.13

* **Duração:** O tempo padrão para a falha é de aproximadamente 45 segundos.40  
* **Período de Graça:** O jogo concede um bônus de 2 segundos se um trem estiver prestes a chegar na estação, criando momentos de alta tensão dramática.41  
* **Recuperação:** Se o número de passageiros cair para 5 ou menos, o temporizador começa a recuar lentamente. Ele não desaparece instantaneamente, o que significa que estações que operam constantemente no limite são muito mais perigosas.39  
* **Impacto das Interchanges:** Uma estação atualizada para Interchange aumenta seu limite para 18 passageiros e acelera a velocidade com que os passageiros entram e saem dos trens.13

## **Diferenciação Paramétrica por Cidade**

Um clone exato deve replicar as nuances estatísticas de cada mapa. As cidades não mudam apenas a arte de fundo; elas alteram as constantes fundamentais do motor de simulação.43

| Cidade | Limite de Estação | Capacidade de Trem | Outras Características |
| :---- | :---- | :---- | :---- |
| Londres | 6 passageiros | 6 passageiros | Configuração padrão; equilíbrio médio de túneis. 18 |
| Paris | 4 passageiros | 6 passageiros | Superpopulação ocorre muito mais rápido. 18 |
| Cairo / Mumbai | 6 passageiros | 4 passageiros | Trens menores exigem maior frequência e frotas maiores. 18 |
| Hong Kong | 6 passageiros | 6 passageiros | Crescimento populacional explosivo; recebe 2 trens por semana. 43 |
| Osaka | 6 passageiros | 6 passageiros | Oferece o Shinkansen (trem de alta velocidade) como upgrade. 18 |
| Melbourne | 6 passageiros | 6 passageiros | Usa bondes que viajam em velocidade constante e lenta. 18 |
| São Paulo | 6 passageiros | 6 passageiros | Passageiros levam mais tempo para embarcar e desembarcar. 44 |

Além dessas variáveis, a distribuição geográfica de estações raras é pré-definida. Em Nova York, as estações quadradas tendem a surgir apenas na ilha de Manhattan, forçando o tráfego dos distritos periféricos para o centro.46 Em Berlim, a escassez crítica de túneis (apenas 1 por semana) torna a travessia do rio Spree o principal desafio estratégico.43

## **Estratégias Emergentes e Comportamento do Usuário**

O planejamento de um clone deve levar em conta como os jogadores de alto nível interagem com o sistema. Mecânicas que podem parecer bugs ou exploits são, na verdade, partes integrantes da experiência Mini Metro.

### **Linhas Fantasma (Ghost Lines)**

Jogadores experientes utilizam slots de linha temporários para aliviar estações críticas. Eles criam uma linha, colocam um trem para recolher os passageiros e deletam a linha imediatamente.16 O trem continuará sua viagem até a próxima estação para descarregar, mas o slot da linha e a locomotiva voltam para o inventário do jogador.16 Um clone que impeça essa manobra falhará em capturar a profundidade estratégica do original.

### **Malhas vs. Loops**

Embora iniciantes prefiram linhas de ponto a ponto, a eficiência máxima é alcançada através de loops ou redes em grade (grid systems).10 Loops permitem que os trens circulem continuamente sem precisar inverter a direção, o que economiza tempo valioso em estações terminais. Além disso, loops com trens em ambas as direções são significativamente mais eficazes em distribuir a carga de passageiros.16

## **Considerações Finais para o Planejamento do Clone**

Para reconstruir Mini Metro com fidelidade, o foco não deve estar na complexidade gráfica, mas na precisão da simulação de fluxo. O sistema deve ser tratado como um motor de otimização de grafos onde cada passageiro é uma unidade de custo e cada locomotiva é uma unidade de rendimento. A elegância do jogo reside na maneira como ele oculta cálculos matemáticos complexos sob uma interface intuitiva e uma sonoplastia relaxante, transformando o caos logístico em uma experiência meditativa de resolução de problemas.1 A implementação bem-sucedida exigirá um equilíbrio rigoroso entre a aleatoriedade procedural e as salvaguardas de design que garantem a jogabilidade contínua.

#### **Referências citadas**

1. Mini Metro and Mini Motorways: The Art of Elegant Constraint Optimization | by John Brandon Elam | Gaming Is Good | Medium, acessado em abril 6, 2026, [https://medium.com/gaming-is-good/mini-metro-and-mini-motorways-the-art-of-elegant-constraint-optimization-2571a32fdfe2](https://medium.com/gaming-is-good/mini-metro-and-mini-motorways-the-art-of-elegant-constraint-optimization-2571a32fdfe2)  
2. Harry Beck's Tube map \- Transport for London, acessado em abril 6, 2026, [https://tfl.gov.uk/corporate/about-tfl/culture-and-heritage/art-and-design/harry-becks-tube-map](https://tfl.gov.uk/corporate/about-tfl/culture-and-heritage/art-and-design/harry-becks-tube-map)  
3. Harry Beck \- More than a Map \- Google Arts & Culture, acessado em abril 6, 2026, [https://artsandculture.google.com/story/harry-beck-more-than-a-map-tfl-archives/IQXRldY73iB1oA?hl=en](https://artsandculture.google.com/story/harry-beck-more-than-a-map-tfl-archives/IQXRldY73iB1oA?hl=en)  
4. Harry Beck's Radical Idea: A Lesson in Disruption | by Pav\_Ka | Bootcamp \- Medium, acessado em abril 6, 2026, [https://medium.com/design-bootcamp/harry-becks-radical-idea-a-lesson-in-disruption-6637fe2ac38e](https://medium.com/design-bootcamp/harry-becks-radical-idea-a-lesson-in-disruption-6637fe2ac38e)  
5. HENRY CHARLES BECK, MATERIAL CULTURE AND THE LONDON TUBE MAP OF 1933, acessado em abril 6, 2026, [https://amodern.net/article/henry-c-beck-material-culture-and-the-london-tube-map-of-1933/](https://amodern.net/article/henry-c-beck-material-culture-and-the-london-tube-map-of-1933/)  
6. championswimmer/metromap.io \- GitHub, acessado em abril 6, 2026, [https://github.com/championswimmer/metromap.io](https://github.com/championswimmer/metromap.io)  
7. AGENTS.md \- championswimmer/metromap-game \- GitHub, acessado em abril 6, 2026, [https://github.com/championswimmer/metromap.io/blob/main/AGENTS.md](https://github.com/championswimmer/metromap.io/blob/main/AGENTS.md)  
8. How to get started with the splines package \- YouTube, acessado em abril 6, 2026, [https://www.youtube.com/watch?v=IJbH5OZa\_is](https://www.youtube.com/watch?v=IJbH5OZa_is)  
9. How to Extrude a Custom Mesh along Splines in Unity | by Lukas Kuppers \- Medium, acessado em abril 6, 2026, [https://medium.com/@lkuppers11/how-to-extrude-a-custom-mesh-along-splines-in-unity-833a97440c1b](https://medium.com/@lkuppers11/how-to-extrude-a-custom-mesh-along-splines-in-unity-833a97440c1b)  
10. We Asked a Transit Planner How to Up Our 'Mini Metro' Game \- VICE, acessado em abril 6, 2026, [https://www.vice.com/en/article/we-asked-a-transit-planner-how-to-up-our-mini-metro-game/](https://www.vice.com/en/article/we-asked-a-transit-planner-how-to-up-our-mini-metro-game/)  
11. What Is a Gameplay Loop? Types of Core Loops Explained, acessado em abril 6, 2026, [https://vsquad.art/blog/what-gameplay-loop-types-core-loops-explained](https://vsquad.art/blog/what-gameplay-loop-types-core-loops-explained)  
12. Mini Metro | Notion, acessado em abril 6, 2026, [https://raghvi.notion.site/Mini-Metro-13f5bea40de540d28593fb10c1a9fe36](https://raghvi.notion.site/Mini-Metro-13f5bea40de540d28593fb10c1a9fe36)  
13. Gameplay \- Mini Metro Wiki \- Fandom, acessado em abril 6, 2026, [https://mini-metro.fandom.com/wiki/Mini\_Metro](https://mini-metro.fandom.com/wiki/Mini_Metro)  
14. Interchange \- Mini Metro Wiki \- Fandom, acessado em abril 6, 2026, [https://mini-metro.fandom.com/wiki/Interchange](https://mini-metro.fandom.com/wiki/Interchange)  
15. Locomotive \- Mini Metro Wiki \- Fandom, acessado em abril 6, 2026, [https://mini-metro.fandom.com/wiki/Locomotive](https://mini-metro.fandom.com/wiki/Locomotive)  
16. Mini Metro \- 5 Beginner Tips to Reach Top 10% on the Leaderboard \- HAMY LABS, acessado em abril 6, 2026, [https://hamy.xyz/blog/2024-03\_mini-metro-tips](https://hamy.xyz/blog/2024-03_mini-metro-tips)  
17. Station \- Mini Metro Wiki \- Fandom, acessado em abril 6, 2026, [https://mini-metro.fandom.com/wiki/Station](https://mini-metro.fandom.com/wiki/Station)  
18. MiniMetro Trivia \- Reddit, acessado em abril 6, 2026, [https://www.reddit.com/r/MiniMetro/comments/7g29c7/minimetro\_trivia/](https://www.reddit.com/r/MiniMetro/comments/7g29c7/minimetro_trivia/)  
19. Visualizing the Theory of Constraints with Mini Metro \- Forte Labs, acessado em abril 6, 2026, [https://fortelabs.com/blog/visualizing-the-theory-of-constraints-with-mini-metro/](https://fortelabs.com/blog/visualizing-the-theory-of-constraints-with-mini-metro/)  
20. What does each shape represent? (If anything) :: Mini Metro General Discussions, acessado em abril 6, 2026, [https://steamcommunity.com/app/287980/discussions/0/458604254470242738/](https://steamcommunity.com/app/287980/discussions/0/458604254470242738/)  
21. Passenger | Mini Metro Wiki \- Fandom, acessado em abril 6, 2026, [https://mini-metro.fandom.com/wiki/Passenger](https://mini-metro.fandom.com/wiki/Passenger)  
22. My "strategy guide" for Mini Metro : r/MiniMetro \- Reddit, acessado em abril 6, 2026, [https://www.reddit.com/r/MiniMetro/comments/ceyt26/my\_strategy\_guide\_for\_mini\_metro/](https://www.reddit.com/r/MiniMetro/comments/ceyt26/my_strategy_guide_for_mini_metro/)  
23. Question: How do passengers choose which station they go to? :: Mini Metro General Discussions \- Steam Community, acessado em abril 6, 2026, [https://steamcommunity.com/app/287980/discussions/0/627456486607812743/](https://steamcommunity.com/app/287980/discussions/0/627456486607812743/)  
24. Postmortem: Dinosaur Polo Club's Mini Metro \- Game Developer, acessado em abril 6, 2026, [https://www.gamedeveloper.com/audio/postmortem-dinosaur-polo-club-s-i-mini-metro-i-](https://www.gamedeveloper.com/audio/postmortem-dinosaur-polo-club-s-i-mini-metro-i-)  
25. Station Type Probability Balancing :: Mini Metro Dyskusje ogólne \- Steam Community, acessado em abril 6, 2026, [https://steamcommunity.com/app/287980/discussions/0/616187203868173967/?l=polish](https://steamcommunity.com/app/287980/discussions/0/616187203868173967/?l=polish)  
26. I'm new to this game. Heard that it's not optimal to have 3 or more of the same shape on the same line consecutively. But I get a spawn like this. Is this normal or am I just unlucky? : r/MiniMetro \- Reddit, acessado em abril 6, 2026, [https://www.reddit.com/r/MiniMetro/comments/1mon2z6/im\_new\_to\_this\_game\_heard\_that\_its\_not\_optimal\_to/](https://www.reddit.com/r/MiniMetro/comments/1mon2z6/im_new_to_this_game_heard_that_its_not_optimal_to/)  
27. What is the algorithm of the spawning of shapes? : r/MiniMetro \- Reddit, acessado em abril 6, 2026, [https://www.reddit.com/r/MiniMetro/comments/1cxstfq/what\_is\_the\_algorithm\_of\_the\_spawning\_of\_shapes/](https://www.reddit.com/r/MiniMetro/comments/1cxstfq/what_is_the_algorithm_of_the_spawning_of_shapes/)  
28. Creating a Mesh \- Catlike Coding, acessado em abril 6, 2026, [https://catlikecoding.com/unity/tutorials/procedural-meshes/creating-a-mesh/](https://catlikecoding.com/unity/tutorials/procedural-meshes/creating-a-mesh/)  
29. Create a tube-shaped mesh along a spline \- Unity \- Manual, acessado em abril 6, 2026, [https://docs.unity3d.com/Packages/com.unity.splines@2.3/manual/extrude-mesh.html](https://docs.unity3d.com/Packages/com.unity.splines@2.3/manual/extrude-mesh.html)  
30. Procedural Meshes for Lines in Unity – code-spot, acessado em abril 6, 2026, [https://www.code-spot.co.za/2020/11/10/procedural-meshes-for-lines-in-unity/](https://www.code-spot.co.za/2020/11/10/procedural-meshes-for-lines-in-unity/)  
31. From Code to Shape: A Beginner's Guide to Unity Mesh Generation | by Prasanth | Medium, acessado em abril 6, 2026, [https://sivakumar-prasanth.medium.com/from-code-to-shape-a-beginners-guide-to-unity-mesh-generation-225cc994e5a6](https://sivakumar-prasanth.medium.com/from-code-to-shape-a-beginners-guide-to-unity-mesh-generation-225cc994e5a6)  
32. Adjusting a line to go around the river : r/MiniMetro \- Reddit, acessado em abril 6, 2026, [https://www.reddit.com/r/MiniMetro/comments/sbw6zp/adjusting\_a\_line\_to\_go\_around\_the\_river/](https://www.reddit.com/r/MiniMetro/comments/sbw6zp/adjusting_a_line_to_go_around_the_river/)  
33. Refinements to dragging lines to stations? :: Mini Metro Discusiones generales \- Steam Community, acessado em abril 6, 2026, [https://steamcommunity.com/app/287980/discussions/0/35220315951327104/?l=latam](https://steamcommunity.com/app/287980/discussions/0/35220315951327104/?l=latam)  
34. Refinements to dragging lines to stations? :: Mini Metro General Discussions, acessado em abril 6, 2026, [https://steamcommunity.com/app/287980/discussions/0/35220315951327104/](https://steamcommunity.com/app/287980/discussions/0/35220315951327104/)  
35. Trains will not pickup riders regardless of space :: Mini Metro General Discussions, acessado em abril 6, 2026, [https://steamcommunity.com/app/287980/discussions/0/667222787665959100/](https://steamcommunity.com/app/287980/discussions/0/667222787665959100/)  
36. Postmortem: Mini Metro \- Disasterpeace, acessado em abril 6, 2026, [https://disasterpeace.com/blog/mini-metro.postmortem](https://disasterpeace.com/blog/mini-metro.postmortem)  
37. The Programmed Music of “Mini Metro” – Interview with Rich Vreeland (Disasterpeace), acessado em abril 6, 2026, [https://designingsound.org/2016/02/18/the-programmed-music-of-mini-metro-interview-with-rich-vreeland-disasterpeace/](https://designingsound.org/2016/02/18/the-programmed-music-of-mini-metro-interview-with-rich-vreeland-disasterpeace/)  
38. Mini Metro (video game) \- Wikipedia, acessado em abril 6, 2026, [https://en.wikipedia.org/wiki/Mini\_Metro\_(video\_game)](https://en.wikipedia.org/wiki/Mini_Metro_\(video_game\))  
39. Normal \- Mini Metro Wiki \- Fandom, acessado em abril 6, 2026, [https://mini-metro.fandom.com/wiki/Normal](https://mini-metro.fandom.com/wiki/Normal)  
40. The Mini-Metro Extensive Guide (beta5) \- Steam Community, acessado em abril 6, 2026, [https://steamcommunity.com/sharedfiles/filedetails/?id=313791555](https://steamcommunity.com/sharedfiles/filedetails/?id=313791555)  
41. How long does it take for a station to overcrowd? :: Mini Metro Discussioni generali, acessado em abril 6, 2026, [https://steamcommunity.com/app/287980/discussions/0/35221584552224188/?l=italian](https://steamcommunity.com/app/287980/discussions/0/35221584552224188/?l=italian)  
42. How long does it take for a station to overcrowd? :: Mini Metro General Discussions, acessado em abril 6, 2026, [https://steamcommunity.com/app/287980/discussions/0/35221584552224188/](https://steamcommunity.com/app/287980/discussions/0/35221584552224188/)  
43. Cities differences? :: Mini Metro General Discussions \- Steam Community, acessado em abril 6, 2026, [https://steamcommunity.com/app/287980/discussions/0/496880503070365589/](https://steamcommunity.com/app/287980/discussions/0/496880503070365589/)  
44. Cities differences? :: Mini Metro General Discussions \- Steam Community, acessado em abril 6, 2026, [https://steamcommunity.com/app/287980/discussions/0/3175603565748281727/](https://steamcommunity.com/app/287980/discussions/0/3175603565748281727/)  
45. City Special Characteristics : r/MiniMetro \- Reddit, acessado em abril 6, 2026, [https://www.reddit.com/r/MiniMetro/comments/1gbp8e8/city\_special\_characteristics/](https://www.reddit.com/r/MiniMetro/comments/1gbp8e8/city_special_characteristics/)  
46. \[Request\] Guide to what makes each city unique : r/MiniMetro \- Reddit, acessado em abril 6, 2026, [https://www.reddit.com/r/MiniMetro/comments/efx037/request\_guide\_to\_what\_makes\_each\_city\_unique/](https://www.reddit.com/r/MiniMetro/comments/efx037/request_guide_to_what_makes_each_city_unique/)  
47. Shinkansen | Mini Metro Wiki \- Fandom, acessado em abril 6, 2026, [https://mini-metro.fandom.com/wiki/Shinkansen](https://mini-metro.fandom.com/wiki/Shinkansen)  
48. Still very new to this game, what am I doing wrong and how can I improve? : r/MiniMetro, acessado em abril 6, 2026, [https://www.reddit.com/r/MiniMetro/comments/1iu8bg7/still\_very\_new\_to\_this\_game\_what\_am\_i\_doing\_wrong/](https://www.reddit.com/r/MiniMetro/comments/1iu8bg7/still_very_new_to_this_game_what_am_i_doing_wrong/)  
49. Gids :: Mini Metro: Basic tips and advanced tips. \- Steam Community, acessado em abril 6, 2026, [https://steamcommunity.com/sharedfiles/filedetails/?l=dutch\&id=606368984](https://steamcommunity.com/sharedfiles/filedetails/?l=dutch&id=606368984)