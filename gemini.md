GEMINI.md — Relaxation Audio Engine 1.0
1. CONTEXTO DO PROJETO

Desenvolver o Relaxation Audio Engine 1.0, um sistema profissional em Python para geração automatizada de áudios de relaxamento, sono, meditação, ASMR, natureza e ambientes cinematográficos.

O sistema deverá combinar:

Síntese sonora procedural.
Composição musical algorítmica.
Progressões harmônicas.
Instrumentos sintetizados e samples reais.
Gravações ambientais.
Processamento digital de sinais (DSP).
Mixagem, espacialização e mastering.

O objetivo é construir um motor flexível, modular, expansível e capaz de produzir áudios originais de alta qualidade, inclusive renderizações de 8 a 10 horas.

Não desenvolver scripts independentes para cada estilo. Todos os estilos devem utilizar a mesma infraestrutura, variando apenas suas configurações e plugins.

2. ARQUIVOS EXISTENTES

O projeto contém diversos scripts Python experimentais que representam versões anteriores dos geradores de áudio.

Esses arquivos devem ser analisados para identificar:

Algoritmos de síntese úteis.
Técnicas de geração de ruído.
Geradores de chuva, trovões e ambientes.
Implementações de instrumentos.
Binaural.
Efeitos DSP.
Mixagem estéreo.
Processamento por chunks.
Gerenciamento de samples externos.

IMPORTANTE:

Os códigos antigos são referências técnicas, não a arquitetura definitiva.

Fazer um rewrite organizado, aproveitando os algoritmos úteis, corrigindo problemas e eliminando duplicações.

Nunca excluir ou sobrescrever os arquivos originais sem autorização.

3. ASSETS DE ÁUDIO

Existe uma pasta audios/ contendo arquivos MP3 de:

Chuva.
Pássaros.
Lareira.
Trovões.
Ruídos ambientais.

Esses arquivos devem ser incorporados ao sistema por meio de um Audio Asset Manager.

Requisitos:

Suportar MP3, WAV, FLAC e OGG.
Converter automaticamente sample rates.
Preservar ou converter canais conforme necessário.
Permitir ajustes individuais de volume.
Permitir seleção determinística por seed.
Permitir loops com crossfade.
Permitir reprodução contínua ou eventos esporádicos.
Evitar repetição excessivamente perceptível.
Não carregar arquivos inteiros de várias horas na memória.

Preservar os arquivos originais.

4. ARQUITETURA OBRIGATÓRIA

Utilizar arquitetura modular baseada em plugins.

Tecnologias:

Python 3.11+
NumPy
SciPy
SoundFile
PyYAML
FFmpeg para decodificação e conversão de formatos quando necessário.
Pytest para testes automatizados.

Arquitetura principal:

Core

Responsável pela orquestração geral.

Componentes:

Engine.
PluginManager.
Scheduler.
Timeline.
Context.
ConfigLoader.
RandomManager.
MusicEngine

Responsável pela composição musical.

Componentes:

ScaleEngine.
HarmonyEngine.
ProgressionEngine.
MelodyEngine.
ArpeggioEngine.
BassEngine.
ArrangementEngine.
Plugins

Organizar plugins em três categorias:

Instruments

Ambient Pad.
Piano.
Harp.
Lute.
Singing Bowl.
Bells.
Bass.
Strings.
Flute.

Ambient

Rain.
Birds.
Fireplace.
Thunder.
Ocean.
Wind.
Forest.
Water.

Textures

White Noise.
Pink Noise.
Brown Noise.
Drone.
Harmonic Drone.
Binaural.
Atmospheric Textures.
Effects
Reverb.
Convolution Reverb.
Delay.
Echo.
Chorus.
Filters.
Saturation.
Stereo Width.
Mixer
Controle individual de volume.
Pan estéreo.
Automação.
Ducking opcional.
Mixagem de camadas.
Controle de dinâmica.
Renderer
Renderização em chunks.
Escrita contínua em disco.
Exportação WAV.
PCM 24-bit.
Metadados.
Relatório técnico.
Mastering
Loudness integrado.
True Peak.
Controle de clipping.
Normalização.
Fade-in e fade-out.
Análise de dinâmica.
5. MUSIC ENGINE — COMPOSIÇÃO HÍBRIDA

A geração musical deve combinar síntese procedural, composição algorítmica e samples reais.

Não utilizar aleatoriedade musical sem controle harmônico.

O sistema deve possuir:

Sistema harmônico
Tonalidades maiores e menores.
Escalas dóricas, lídias, mixolídias, pentatônicas e outras.
Progressões harmônicas configuráveis.
Acordes com extensões.
Inversões.
Condução de vozes.
Transições harmônicas suaves.
Modulação opcional.
Sistema melódico
Geração de melodias coerentes com a tonalidade.
Notas de repouso.
Variações melódicas.
Arpejos.
Intervalos controlados.
Densidade configurável.
Silêncios musicais.
Instrumentos

Cada instrumento deverá possuir timbre, envelope, articulação, volume e efeitos próprios.

Evolução musical

O áudio deve evoluir naturalmente durante sua duração.

Permitir configurar:

Introdução.
Desenvolvimento.
Variação harmônica.
Entrada e saída de instrumentos.
Redução gradual da densidade musical.
Encerramento suave.

Para Deep Sleep, permitir composições extremamente discretas ou sem melodias.

A música não deve apresentar mudanças bruscas ou elementos excessivamente chamativos.

6. SISTEMA DE PLUGINS

Todos os plugins devem seguir um contrato comum.

Cada plugin deverá possuir:

Identificador único.
Categoria.
Configuração própria.
Método de preparação.
Método de renderização.
Estado interno quando necessário.
Controle de ativação.
Reset de estado.

O método de renderização deverá trabalhar com intervalos de amostras:

render(start_sample, num_samples)

Os plugins devem funcionar independentemente e não depender de outros plugins específicos.

O Core deverá gerenciar o carregamento e a execução.

Adicionar novos instrumentos, ambientes ou efeitos não deverá exigir modificações no núcleo.

7. PROCESSAMENTO DE ÁUDIO

Configuração padrão:

Sample rate: 44.100 Hz.
Canais: 2.
Processamento interno: float32.
Exportação: PCM 24-bit.
Chunk padrão: 10 segundos.

Requisitos obrigatórios:

Processamento por chunks.
Estado dos filtros preservado entre chunks.
Reverb com caudas preservadas.
Continuidade de fase quando aplicável.
Eventos baseados em tempo absoluto.
Crossfades nos loops.
Ausência de cliques nas fronteiras dos chunks.
Memória RAM aproximadamente constante.
Renderizações determinísticas.

Evitar armazenar o áudio completo na memória.

Utilizar escrita contínua em disco e processamento em múltiplas passagens quando necessário.

Não normalizar individualmente cada chunk.

8. RANDOMIZAÇÃO E REPRODUTIBILIDADE

Implementar RandomManager centralizado.

Cada projeto terá uma seed principal e cada plugin deverá receber uma seed derivada independente.

Requisitos:

Mesma configuração e seed devem produzir resultados reproduzíveis.
Plugins não devem compartilhar estados globais de randomização.
Eventos devem ser independentes da ordem de processamento dos chunks.
Seeds diferentes devem produzir variações musicais e ambientais.
Renderizar com tamanhos de chunks diferentes não deve alterar a composição.
9. PERFIS DE ÁUDIO

Utilizar YAML para configurar os estilos.

Criar os seguintes perfis iniciais:

Deep Sleep.
Meditation.
Relaxation.
Nature Sounds.
ASMR.
Fantasy Tavern.
Medieval Ambient.
Cinematic Ambient.

Cada perfil deverá controlar:

Duração.
Tonalidade.
Escala.
Progressão harmônica.
Instrumentos.
Plugins ambientais.
Texturas.
Efeitos.
Volumes.
Automação.
Mastering.

Os perfis devem ser configuráveis sem modificar código Python.

10. MASTERING E QUALIDADE

Implementar um pipeline de mastering confiável.

Utilizar medição compatível com EBU R128 para loudness integrado e medição de true peak.

Não considerar uma simples estimativa RMS como medição LUFS profissional.

Permitir definir alvos por perfil.

Parâmetros configuráveis:

Target LUFS.
True Peak máximo.
Ganho.
Limitação.
Dinâmica.
Fade-in.
Fade-out.

Evitar compressão excessiva e preservar a dinâmica natural.

Não aplicar um único alvo de loudness indiscriminadamente a todos os estilos.

11. INTERFACE CLI

Permitir execução através de comandos.

Exemplos desejados:

python main.py render --profile deep_sleep --duration 28800 --seed 123

python main.py render --profile fantasy_tavern --duration 3600 --seed 456

python main.py render --profile meditation --duration 1800 --seed 789

Também disponibilizar:

python main.py list-profiles

python main.py list-plugins

python main.py validate

python main.py info

12. ESTRUTURA DE DIRETÓRIOS

Organizar o projeto inicialmente desta maneira:

relaxation-audio-engine/

main.py
requirements.txt
pyproject.toml
GEMINI.md
README.md
config/
engine.yaml
profiles/
core/
engine.py
plugin_manager.py
scheduler.py
timeline.py
context.py
config_loader.py
random_manager.py
music/
harmony.py
progression.py
scale.py
composer.py
melody.py
rhythm.py
arrangement.py
plugins/
instruments/
ambient/
textures/
effects/
assets/
audio/
samples/
impulses/
mixer/
renderer/
tests/
docs/
output/

Não mover os arquivos originais da pasta audios/ sem autorização.

13. TESTES OBRIGATÓRIOS

Criar testes automatizados para:

Inicialização do Core.
Carregamento dos plugins.
Validação de configurações YAML.
Reprodutibilidade por seed.
Continuidade entre chunks.
Ausência de clipping.
Medição de loudness.
Crossfades.
Mixagem estéreo.
Duração final.
Consumo de memória.
Compatibilidade entre plugins.

Antes de renderizar várias horas, validar o sistema com áudios de 30 segundos, 2 minutos e 10 minutos.

14. REGRAS DE DESENVOLVIMENTO
Não concentrar funcionalidades em um único arquivo.
Não duplicar algoritmos sem justificativa.
Não criar funcionalidades fictícias ou placeholders apresentados como concluídos.
Não ignorar erros silenciosamente.
Utilizar type hints e documentação.
Manter separação entre Core, MusicEngine, Plugins, Mixer e Renderer.
Priorizar qualidade sonora e estabilidade.
Validar os módulos antes de integrá-los.
Não alterar arquivos originais sem autorização.
Manter README e documentação atualizados.
Registrar decisões arquiteturais importantes.
Não considerar uma etapa concluída sem testes.
15. PLANO DE IMPLEMENTAÇÃO

Desenvolver em etapas:

Fase 1 — Fundação

Analisar os scripts existentes.
Analisar os assets.
Criar estrutura do projeto.
Implementar configurações.
Implementar Core.
Implementar PluginManager.
Implementar RandomManager.
Implementar Scheduler.
Implementar Renderer básico.

Fase 2 — MusicEngine

Implementar escalas.
Implementar acordes.
Implementar progressões.
Implementar melodias.
Implementar arpejos.
Implementar instrumentos iniciais.

Fase 3 — Ambient Engine

Implementar Audio Asset Manager.
Integrar MP3 existentes.
Implementar geradores ambientais.
Implementar texturas.
Implementar binaural opcional.

Fase 4 — DSP e Mixer

Implementar efeitos.
Implementar mixagem.
Implementar automação.
Implementar espacialização.

Fase 5 — Mastering

Implementar análise de loudness.
Implementar true peak.
Implementar exportação PCM 24-bit.
Gerar relatórios.

Fase 6 — Perfis e validação

Criar os oito perfis iniciais.
Realizar testes de integração.
Testar renderizações longas.
Documentar limitações e resultados.
16. INSTRUÇÃO FINAL AO AGENTE

Você deve agir como engenheiro responsável pelo projeto, não como gerador de exemplos de código.

Antes de implementar:

Analise todos os scripts Python disponíveis.
Analise a pasta audios/.
Identifique funcionalidades aproveitáveis.
Identifique problemas técnicos.
Apresente o diagnóstico e o plano de execução.

Depois comece a implementação incremental da versão 1.0.

Não tente implementar todo o projeto em um único arquivo ou em uma única operação.

Mantenha um arquivo IMPLEMENTATION_PLAN.md com:

Etapas concluídas.
Etapas pendentes.
Testes executados.
Resultados.
Problemas conhecidos.
Próximas tarefas.

A prioridade absoluta é construir um motor de áudio modular, musicalmente coerente, com qualidade profissional, renderização estável e capacidade de expansão.

Toda funcionalidade deve ser implementada, testada e documentada antes de ser marcada como concluída.