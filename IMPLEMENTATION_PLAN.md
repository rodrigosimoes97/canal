# IMPLEMENTATION_PLAN.md — Relaxation Audio Engine 1.0

Este documento registra o estado atual, diagnóstico técnico, planejamento e progresso do desenvolvimento do **Relaxation Audio Engine 1.0**, conforme as diretrizes do `GEMINI.md`.

---

## 1. Diagnóstico Técnico Inicial

### 1.1 Análise dos Assets (`audios/`)
- `chuva.mp3`: 48.000 Hz, 2 canais (estéreo), 765.10 s (~12.7 min)
- `lareira.mp3`: 44.100 Hz, 2 canais (estéreo), 631.03 s (~10.5 min)
- `passaro.mp3`: 44.100 Hz, 1 canal (mono), 2.56 s (evento one-shot pontual)
- `passaro2.mp3`: 44.100 Hz, 2 canais (estéreo), 17.60 s (evento ambiental curto)
- `ruido.mp3`: 24.000 Hz, 2 canais (estéreo), 9.00 s (sample rate não-padrão)
- `trovao.mp3`: 44.100 Hz, 2 canais (estéreo), 157.48 s (~2.6 min)
- `vento.mp3`: 48.000 Hz, 2 canais (estéreo), 18.41 s

**Desafios Técnicos Identificados nos Assets e Soluções Adotadas:**
1. **Heterogeneidade de Sample Rates:** Resolvido no `AudioAssetManager` através de resampling polifásico de fase linear (`scipy.signal.resample_poly`), convertendo 24 kHz e 48 kHz para 44.1 kHz sem aliasing e sem custo excessivo de RAM.
2. **Heterogeneidade de Canais:** Conversão automática mono -> estéreo garantindo compatibilidade uniforme em todos os plugins.
3. **Looping Contínuo sem Cliques:** Implementado buffer cíclico com crossfade equal-power senoidal pré-calculado na cauda do arquivo. A leitura por indexação modular direta $O(1)$ elimina qualquer descontinuidade.
4. **Consumo de Memória:** O streaming contínuo por chunks lê do cache polifásico sem reter matrizes duplicadas em memória.

### 1.2 Análise dos Scripts Legados (`base/`)
Scripts avaliados: `zen_automator.py`, `zen_automator_v2.py`, `zen_automator_v3.py`, `zen_automator_v4.py`, `zen_automator_v5.py`, `zen_automator_v6.py`, `zen_claude.py`, `zen_pro_relax.py`, `zen_tavern_asmr.py`, `zen_teste.py`.

---

## 2. Dependências do Sistema

### 2.1 Instaladas e Ativas no Ambiente (`venv`):
- `python`: 3.13.13
- `numpy`: 2.4.4
- `scipy`: 1.17.1
- `soundfile`: 0.13.1
- `cffi`: 2.0.0
- `pyyaml`: 6.0.3
- `pyloudnorm`: 0.2.0
- `pytest`: 9.2.0
- `ffmpeg`: 2026-07-16 (disponível no PATH do Windows)

---

## 3. Estado das Etapas de Implementação

### Fase 1 — Fundação e Core Engine [CONCLUÍDA]
- [x] Configurar dependências e empacotamento (`requirements.txt`, `pyproject.toml`).
- [x] Criar configuração padrão do motor em `config/engine.yaml`.
- [x] Implementar `core/context.py` com cálculo de tempo/amostra e limites de chunking.
- [x] Implementar `core/random_manager.py` com geradores PCG64 desacoplados e sementes determinísticas via hash SHA-256.
- [x] Implementar `core/config_loader.py` com carregamento YAML e validação de schema.
- [x] Implementar `plugins/base.py` com contrato `BasePlugin`, categorias e metadados.
- [x] Implementar `core/timeline.py` e `core/scheduler.py` para agendamento temporal de eventos e fatiamento sem descontinuidades.
- [x] Implementar `core/plugin_manager.py` para registro dinâmico e ciclo de vida de plugins.
- [x] Implementar `core/engine.py` como orquestrador mestre integrando contexto, PRNGs, scheduler e plugins.
- [x] Implementar `renderer/chunk_writer.py` para escrita contínua PCM 24-bit em streaming direto em disco com RAM $O(1)$.
- [x] Implementar `renderer/renderer.py` com suporte a callbacks de progresso.
- [x] Testes unitários da Fase 1 aprovados (12 testes).

### Fase 2 — MusicEngine e Instrumentos [CONCLUÍDA]
- [x] Implementar `music/scale.py` (escalas Pentatônica, Lídia, Dórica, Mixolídia, Maior, Menor Natural/Harmônica, afinações Solfeggio 432 Hz).
- [x] Implementar `music/harmony.py` (tríades, tétrades, extensões add9/maj7, inversões, condução suave de vozes `smooth_voice_leading`).
- [x] Implementar `music/progression.py` (sequenciamento de graus harmônicos determinísticos com voicings encadeados).
- [x] Implementar `music/rhythm.py` (divisão rítmica, durações relativas de figuras, BPM e conversões tempo/compasso).
- [x] Implementar `music/melody.py` (frases musicais com motivo, variação, notas de repouso harmônicas, arpejos e silêncios estruturais).
- [x] Implementar `music/arrangement.py` (curva de densidade dinâmica macro-temporal: intro, development, deep_flow, wind_down, outro).
- [x] Implementar `music/composer.py` (fachada integradora conectando contexto, escalas, progressão e arranjo).
- [x] Implementar os 9 instrumentos sintetizados em `plugins/instruments/`:
  - [x] `ambient_pad.py`, `piano.py`, `harp.py`, `lute.py`, `singing_bowl.py`, `bells.py`, `bass.py`, `strings.py`, `flute.py`.
- [x] Testes automatizados da Fase 2 aprovados (18 testes).

### Fase 3 — Ambient Engine e Texturas [CONCLUÍDA]
- [x] Implementar `assets/asset_manager.py`:
  - [x] Indexação de arquivos suportados (.mp3, .wav, .flac, .ogg).
  - [x] Resampling polifásico exato via `scipy.signal.resample_poly` para 44.1 kHz.
  - [x] Conversão mono -> estéreo preservando balanço.
  - [x] Buffers de loop com crossfade circular equal-power sem cliques nas emendas.
  - [x] Cache compartilhado em memória para performance máxima de streaming.
- [x] Integrar os arquivos existentes de `audios/` preservando os originais intactos.
- [x] Implementar os 8 plugins ambientais em `plugins/ambient/`:
  - [x] `rain.py`: chuva híbrida (amostra real `chuva.mp3` + gotas procedurais filtradas).
  - [x] `birds.py`: cantos de pássaros esparsos (`passaro.mp3`, `passaro2.mp3`) com panning estéreo aleatório.
  - [x] `fireplace.py`: lareira contínua (`lareira.mp3`) com estalos procedurais Poisson de madeira.
  - [x] `thunder.py`: trovões distantes e rumble grave sub-grave (< 120 Hz).
  - [x] `ocean.py`: ondas do mar e respiração oceânica procedural com LFO senoidal e arrebentação estéreo.
  - [x] `wind.py`: vento contínuo (`vento.mp3`) combinado com rajadas procedurais.
  - [x] `forest.py`: ambiente de bosque e farfalhar de folhas com filtro passa-faixa contínuo.
  - [x] `water.py`: riacho cristalino e borbulhamento suave contínuo.
- [x] Implementar os plugins de textura em `plugins/textures/`:
  - [x] `noise.py`: ruído branco, rosa (filtro Paul Kellet) e marrom (integrador leaky de 1 polo sem deriva).
  - [x] `binaural.py`: batimentos binaurais reais estéreo isolados (Delta 2.5Hz, Theta 6.0Hz, Alpha 10.0Hz).
  - [x] `drone.py`: drone textural contínuo com modulação lenta de fase e sub-grave.
  - [x] `harmonic_drone.py`: drone harmônico afinado com fundamentais e quintas da progressão ativa.
- [x] Criar suíte de testes `tests/test_assets.py` com 20 casos de teste aprovados.
- [x] Gerar arquivo real WAV em `output/ambient_texture_preview_30s.wav` (44.100 Hz, 2 canais, PCM 24-bit).

### Fase 4 — DSP, Efeitos e Mixer [CONCLUÍDA]
- [x] Implementar efeitos DSP profissionais com preservação contínua de estados IIR/FIR entre chunks:
  - [x] `plugins/effects/filter.py`: Filtros Butterworth de 2ª a 4ª ordem (Low-pass, High-pass, Band-pass) com estados `sos` preservados via `scipy.signal.sosfilt`.
  - [x] `plugins/effects/delay.py`: Delay estéreo independente e Ping-Pong com delay lines circulares persistentes e amortecimento passa-baixa de feedback.
  - [x] `plugins/effects/reverb.py`: Reverb algorítmico estilo Schroeder/Freeverb com 8 comb filters paralelos e 4 all-pass filters em cascata com caudas preservadas entre chunks.
  - [x] `plugins/effects/convolution.py`: Convolução por blocos particionados overlap-add (OLA) em domínio de frequência (FFT) para respostas de impulso com caudas persistentes.
  - [x] `plugins/effects/saturation.py`: Saturação suave não-linear baseada em $\tanh$ ou sigmoid algébrico com ganho de entrada/saída e filtro DC-Blocker IIR de 1 polo contínuo.
  - [x] `plugins/effects/spatializer.py`: Posicionamento espacial com Pan estéreo em lei de potência constante (-3dB equal-power law) e processamento Mid/Side para ajuste de abertura estéreo.
- [x] Implementar sistema modular de canais e mixagem:
  - [x] `mixer/channel.py`: Canais independentes com controle de volume em dB, pan estéreo, mute, solo, inserts de efeitos encadeados, envios auxiliares (Sends) pré/pós fader e automação suave por interpolação de breakpoints.
  - [x] `mixer/mixer.py`: Console de mixagem com barramento mestre (Master Bus), barramentos auxiliares (Aux Buses para Reverb/Delay), soma em precisão estendida (float64) para prevenir ruído de truncamento e headroom de acumulação, limitador suave mestre e Sidechain Ducking configurável com detector envelope (Attack/Release contínuos entre blocos).
- [x] Integrar o `Mixer` ao `core/engine.py` mantendo streaming direto em disco com RAM $O(1)$.
- [x] Implementar suíte de testes dedicados em `tests/test_dsp_continuity.py` (13 testes aprovados).
- [x] Gerar e validar 3 amostras WAV preliminares em `output/` (`sample1_music_reverb_delay.wav`, `sample2_ambient_spatial.wav`, `sample3_full_mix_production.wav`).

### Fase 4.1 — Refinamento Sonoro e Direção Artística [CONCLUÍDA]
- [x] Diagnóstico acústico das amostras preliminares: identificada superioridade da biofonia natural e necessidade de remodelagem física dos instrumentos e subordinação da música à ambiência.
- [x] Refatoração da composição e fraseado no `MusicEngine`:
  - [x] `music/melody.py`: Implementado gerador de gestos esparsos e contemplativos (`generate_sparse_gestures`) com notas raras a cada 20–40 segundos, longos silêncios meditativos, rubato expressivo e eliminações de arpejos quantizados mecânicos.
  - [x] `music/composer.py`: Duração estendida de acordes (24 a 32 beats / ~30 a 45 segundos por acorde) para transições harmônicas ultra-lentas e flutuantes.
- [x] Modelagem físico-acústica e tímbrica dos instrumentos sintetizados:
  - [x] `plugins/instruments/piano.py`: Transformado em *Felt Piano* aveludado e acolhedor (inarmonicidade real de cordas com coeficiente de rigidez $B=0.00028$, triplo uníssono estéreo com micro-detuning $\pm 0.32$ Hz gerando batimento acústico natural, ataque de feltro macio sem estridência, ressonância profunda de tampo harmônico e decaimento exponencial seletivo de agudos).
  - [x] `plugins/instruments/flute.py`: Transformado em *Flauta de Bambu / Shakuhachi* meditativa (turbulência acústica de sopro via ruído passa-faixa sintonizado, ataque com sopro de ar, micro-instabilidade natural de respiração lenta em substituição ao vibrato senoidal mecânico e queda expressiva de afinação no release).
  - [x] `plugins/instruments/ambient_pad.py`: Transformado em *Warm Analog Pad* (filtro IIR passa-baixa analógico contínuo cortando estridência digital em 480 Hz, saturação quente de fita magnética e LFOs assíncronos de respiração espacial).
  - [x] `plugins/effects/reverb.py`: Adicionado amortecimento escuro passa-baixa contínuo (*Dark Warm Reverb*) para submergir instrumentos na névoa sonora sem sibilância digital.
- [x] Rebalanceamento da hierarquia da mesa de mixagem: ambiência natural acústica em 1º plano, texturas harmônicas aveludadas em 2º plano e gestos instrumentais esparsos em 3º plano auditivo distante.
- [x] Implementação de testes dedicados em `tests/test_sound_refinement.py` (7 testes cobrindo inarmonicidade, turbulência de ar, filtragem analógica de corte, gestos esparsos e geração de 3 produções refinadas).
- [x] Geração e validação das 3 amostras WAV refinadas de produção em `output/` (PCM 24-bit, 44.1 kHz, estéreo):
  - `sample1_refined_ambient_music.wav` (35s): Felt Piano esparso e aveludado, Pad quente e chuva suave envolta em Dark Reverb.
  - `sample2_refined_nature_zen.wav` (35s): Ambiência Zen com Floresta, Chuva e Lareira dominante, acompanhada por sopros meditativos de Flauta de Bambu e Tigela Tibetana.
  - `sample3_refined_deep_relaxation.wav` (45s): Produção completa de relaxamento profundo (Deep Sleep / Meditation) com leito acústico natural, batimentos binaurais Theta (6 Hz) e harmonia acolhedora flutuando ao fundo.

### Fase 5 — Mastering e Conformidade EBU R128 [PENDENTE]
- [ ] Implementar `mastering/loudness.py` (medição ITU-R BS.1770-4 / EBU R128 com K-weighting e duplo gating).
- [ ] Implementar `mastering/true_peak.py` (oversampling 4x para detecção de picos interamostrais).
- [ ] Implementar arquitetura de dois passes com RAM constante (`mastering/master.py`).
- [ ] Criar e executar testes `tests/test_loudness.py`.

### Fase 6 — Perfis Declarativos, CLI e Validação Final [PENDENTE]
- [ ] Criar os 8 perfis YAML em `config/profiles/`.
- [ ] Implementar CLI completa em `main.py`.
- [ ] Executar validações práticas de 30s, 2min e 10min.
- [ ] Criar `README.md` completo.

---

## 4. Histórico de Testes e Resultados

| Data / Execução | Suíte | Casos de Teste | Status | Cobertura / Observação |
| :--- | :--- | :--- | :--- | :--- |
| **05/11/2026** | `pytest tests/` | 12 testes | **100% Aprovado** | Validação inicial da Fase 1 (Core Engine, Streaming, RandomManager). |
| **05/11/2026** | `pytest tests/` | 30 testes | **100% Aprovado** | Validação completa da Fase 1 + Fase 2: Escalas, Frequências Solfeggio, Acordes, Inversões, Voice Leading, Progressões, Melodia, Continuidade exata entre chunks na fronteira, Validação individual dos 9 instrumentos e Integração com o Core Engine. |
| **05/11/2026** | `pytest tests/` | 50 testes | **100% Aprovado** | Validação da Fase 1 + 2 + 3: Resampling polifásico exato (24k/48k -> 44.1k), conversão mono/estéreo, loops circulares equal-power sem descontinuidade, validação dos 8 plugins ambientais, 4 plugins de textura (incluindo Binaural Beats com isolamento de canais e ruídos calibrados) e geração de áudio WAV 30s PCM 24-bit. |
| **05/11/2026** | `pytest tests/` | 63 testes | **100% Aprovado (63 passed in 160.03s)** | Validação da Fase 1 + 2 + 3 + 4: Continuidade IIR de filtros LP/HP/BP entre blocos, persistência de caudas de delay e reverb Freeverb, convolução particionada OLA, saturação com DC-blocker, equal-power pan law e largura Mid/Side, automações de canal e sidechain ducking contínuo no Mixer, além de renderização técnica de 3 produções em disco. |
| **05/11/2026** | `pytest tests/` | 70 testes | **100% Aprovado (70 passed)** | Validação da Fase 4.1 (Refinamento Sonoro e Direção Artística): Modelagem acústica de felt piano com inarmonicidade e uníssono triplo, flauta de bambu orgânica com sopro passa-faixa e respiração lenta sem vibrato mecânico, ambient pad analógico com filtro passa-baixa aveludado (480 Hz), dark warm reverb, gerador de gestos esparsos contemplativos com rubato, e geração de 3 novas produções refinadas em PCM 24-bit. |

---

## 5. Limitações Técnicas e Considerações da Fase 4
- **Preservação de Caudas DSP:** O encerramento de notas e sons pontuais propaga caudas de reverberação e eco além da timeline principal. O motor permite estender os blocos de renderização (ring-out tail) para que a cauda de reverb/delay não seja truncada abruptamente.
- **Soma e Headroom:** A soma de múltiplos canais com saturação procedural em canais individuais e barramento mestre utiliza ponto flutuante de 64 bits para total transparência sonora, mantendo headroom seguro (> 7 dB) antes do estágio de masterização.
- **Transição para a Fase 5:** O mixer entrega o sinal estéreo pronto e dinamicamente balanceado. A **Fase 5 (Mastering)** receberá esse áudio para medição em conformidade estrita com a norma EBU R128 / ITU-R BS.1770-4 (LUFS integrado e True Peak 4x oversampled), normalizando o ganho de acordo com o perfil sem clipping.
