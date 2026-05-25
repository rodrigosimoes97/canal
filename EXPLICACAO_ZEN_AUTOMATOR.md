# Documentação Detalhada: zen_automator.py

Este documento explica o funcionamento do script `zen_automator.py`, que é uma ferramenta para geração procedural de áudio relaxante (ambientes "Zen").

---

## 1. Visão Geral
O script utiliza Processamento Digital de Sinais (DSP) para criar uma paisagem sonora composta por:
- **Ruído Marrom (Brown Noise):** Um som profundo e suave que lembra chuva ou vento forte.
- **Batidas Binaurais:** Frequências que ajudam no relaxamento cerebral.
- **Pads Musicais:** Sons de sintetizador suaves que criam uma base harmônica.
- **Melodias Aleatórias:** Notas esporádicas que trazem variação ao áudio.

---

## 2. Dependências
Para rodar este código, você precisa de:
- `numpy`: Para cálculos matemáticos e manipulação de arrays de áudio.
- `soundfile`: Para salvar o arquivo final em formato `.wav`.
- `scipy`: Para filtros de áudio (Butterworth).

---

## 3. Explicação das Funções

### Seção: Utilidades DSP
Estas funções moldam a "textura" do som.

- **`soft_saturation(x, drive=1.2)`**:
    - **O que faz:** Aplica uma distorção suave usando a função tangente hiperbólica (`tanh`).
    - **Efeito:** Dá um "calor analógico" ao som, impedindo que ele soe digitalmente "frio" ou "picado".
- **`simple_reverb(signal, decay=0.4, delay_ms=120)`**:
    - **O que faz:** Pega o som original e o repete com um pequeno atraso e volume reduzido.
    - **Efeito:** Simula o som ecoando em uma sala grande, dando profundidade.
- **`lowpass(data, cutoff=600)`**:
    - **O que faz:** Remove frequências acima de 600Hz.
    - **Efeito:** Torna o som mais "abafado" e relaxante, removendo chiados agudos.
- **`highcut(data, cutoff=40)`**:
    - **O que faz:** Remove frequências muito graves (abaixo de 40Hz) que podem causar ruído indesejado em subwoofers.

### Seção: Geração de Som
- **`generate_brown_noise(n)`**:
    - Cria ruído branco e aplica uma integração cumulativa para transformá-lo em ruído marrom. Filtra o resultado para garantir que seja suave.
- **`apply_binaural(n, base_freq=180, beat_freq=4)`**:
    - Gera dois tons senoidais ligeiramente diferentes: um de 180Hz (ouvido esquerdo) e outro de 184Hz (ouvido direito). A diferença de 4Hz é o "batimento" percebido pelo cérebro.

### Seção: Motor Musical (`MusicEngine`)
Esta classe é o coração da melodia.
- **`smooth_pad(t, freq)`**:
    - Cria um acorde usando três osciladores levemente desafinados (para soar mais rico e "vivo"). Usa um envelope exponencial para o som sumir suavemente.
- **`generate(n, root=110)`**:
    - **Harmonia:** Escolhe frequências baseadas na nota fundamental (110Hz).
    - **Melodia:** A cada 6 segundos, escolhe uma nota aleatória da escala e toca por 4 segundos com um volume baixo.

### Seção: Renderização Final
- **`render(output, duration)`**:
    - É o "maestro" que junta tudo.
    - Cria o ruído, a música e as batidas binaurais.
    - Faz o "mix" (mistura) de tudo.
    - Aplica o Reverb e a Saturação.
    - Aplica um **Fade-in** e **Fade-out** de 8 segundos no início e no fim para o som não começar/parar de forma brusca.
    - **Normalização:** Garante que o volume final esteja no nível máximo sem distorcer.

---

## 4. Como Modificar o Código

Aqui estão algumas ideias de como você pode personalizar o som:

### A. Mudar a "cor" do ruído
No `render`, você pode aumentar ou diminuir a presença do ruído marrom:
```python
noise = generate_brown_noise(n) * 0.25  # Mais presente (original era 0.15)
```

### B. Mudar o tom da música
Na função `main` ou no `render`, você pode alterar a nota fundamental da classe `MusicEngine`:
```python
music = MusicEngine.generate(n, root=80) # Som mais grave/profundo
```

### C. Alterar a velocidade da melodia
No `MusicEngine.generate`, mude o `step`:
```python
step = int(SAMPLE_RATE * 12) # Notas melódicas ocorrem a cada 12 segundos (mais lento)
```

### D. Aumentar o Reverb (Eco)
No `render`, altere o `decay`:
```python
mix = simple_reverb(mix, decay=0.6) # Som muito mais "espacial" e etéreo
```

### E. Mudar a frequência binaural
Se quiser foco em vez de relaxamento profundo, mude o `beat_freq`:
```python
bl, br = apply_binaural(n, beat_freq=10) # 10Hz está na faixa Alpha (foco leve)
```

---

## 5. Como Executar
No terminal, você pode passar parâmetros:
```bash
python zen_automator.py --duration 300 --output medidacao_5min.wav
```
- `--duration`: Duração em segundos (padrão é 120).
- `--output`: Nome do arquivo de saída (padrão é `zen.wav`).
