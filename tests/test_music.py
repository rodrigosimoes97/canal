"""
Suíte de Testes Automatizados para a Fase 2: MusicEngine e Instrumentos.
"""
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from core.context import RenderContext
from core.random_manager import RandomManager
from core.engine import Engine
from renderer.renderer import Renderer

from music.scale import (
    Scale,
    ScaleType,
    midi_to_freq,
    midi_to_note,
    note_to_freq,
    note_to_midi,
)
from music.harmony import (
    Chord,
    ChordQuality,
    smooth_voice_leading,
)
from music.progression import (
    ProgressionEngine,
    STANDARD_PROGRESSIONS,
)
from music.rhythm import (
    RhythmEngine,
    NoteDuration,
)
from music.melody import (
    MelodyEngine,
    MelodicNote,
)
from music.arrangement import (
    ArrangementEngine,
    ArrangementStage,
)
from music.composer import Composer

from plugins.instruments import (
    AmbientPad,
    Piano,
    Harp,
    Lute,
    SingingBowl,
    Bells,
    Bass,
    Strings,
    Flute,
    ALL_INSTRUMENTS,
)


# ---------------------------------------------------------------------------
# 1. Teoria musical e frequências
# ---------------------------------------------------------------------------
def test_scale_and_frequencies():
    # Validações fundamentais: C4 = MIDI 60, A4 = MIDI 69 = 440.0 Hz
    assert note_to_midi("C", 4) == 60
    assert note_to_midi("A", 4) == 69
    assert pytest.approx(midi_to_freq(69, 440.0), 1e-4) == 440.0
    assert pytest.approx(note_to_freq("A", 4, 440.0), 1e-4) == 440.0

    # Afinação Solfeggio 432 Hz
    assert pytest.approx(midi_to_freq(69, 432.0), 1e-4) == 432.0

    # Escala Maior (C Major: C, D, E, F, G, A, B)
    scale_major = Scale(root_note="C", scale_type=ScaleType.MAJOR)
    notes = scale_major.get_note_names()
    assert notes == ["C", "D", "E", "F", "G", "A", "B"]

    # Modo Lídio (C Lydian: 4ª aumentada F#)
    scale_lydian = Scale(root_note="C", scale_type=ScaleType.LYDIAN)
    assert "F#" in scale_lydian.get_note_names()

    # Modo Dórico (D Dorian: D, E, F, G, A, B, C)
    scale_dorian = Scale(root_note="D", scale_type=ScaleType.DORIAN)
    assert scale_dorian.get_note_names() == ["D", "E", "F", "G", "A", "B", "C"]

    # Graus da escala
    assert scale_major.get_degree_midi(1, octave=4) == 60 # C4
    assert scale_major.get_degree_midi(3, octave=4) == 64 # E4
    assert scale_major.get_degree_midi(5, octave=4) == 67 # G4
    assert scale_major.get_degree_midi(8, octave=4) == 72 # C5 (oitava)


# ---------------------------------------------------------------------------
# 2. Formação e inversão de acordes
# ---------------------------------------------------------------------------
def test_chord_formation_and_inversions():
    # C Major fundamental: C4 (60), E4 (64), G4 (67)
    c_maj = Chord(root_note="C", quality=ChordQuality.MAJOR, octave=4)
    assert c_maj.get_midi_notes() == [60, 64, 67]

    # 1ª Inversão: E4 (64), G4 (67), C5 (72)
    c_inv1 = c_maj.invert(1)
    assert c_inv1.get_midi_notes() == [64, 67, 72]

    # 2ª Inversão: G4 (67), C5 (72), E5 (76)
    c_inv2 = c_maj.invert(2)
    assert c_inv2.get_midi_notes() == [67, 72, 76]

    # Extensão: Cadd9 -> C4 (60), E4 (64), G4 (67), D5 (74)
    c_add9 = Chord(root_note="C", quality=ChordQuality.ADD9, octave=4)
    assert c_add9.get_midi_notes() == [60, 64, 67, 74]

    # Acorde com baixo (Slash chord): C/G
    c_slash_g = Chord(root_note="C", quality=ChordQuality.MAJOR, octave=4, bass_note="G")
    notes = c_slash_g.get_midi_notes()
    # O baixo (G) deve ser a nota mais grave
    assert notes[0] % 12 == 7 # G
    assert notes[0] < notes[1]


# ---------------------------------------------------------------------------
# 3. Voice leading (condução suave de vozes)
# ---------------------------------------------------------------------------
def test_voice_leading_smoothness():
    # Transição clássica C -> F
    # C: [60, 64, 67] (C4, E4, G4)
    # F fundamental seria [65, 69, 72] (deslocamento de +5, +5, +5 = 15 semitons)
    # Voice leading suave para F: 2ª inversão [60, 65, 69] (C4, F4, A4) -> deslocamento de (0 + 1 + 2 = 3 semitons!)
    c_chord = Chord(root_note="C", quality=ChordQuality.MAJOR, octave=4)
    f_chord = Chord(root_note="F", quality=ChordQuality.MAJOR, octave=4)

    c_voicing = c_chord.get_midi_notes() # [60, 64, 67]
    f_voiced = smooth_voice_leading(c_voicing, f_chord)

    # Nota comum C4 (60) deve ser preservada
    assert 60 in f_voiced
    # Deslocamento total deve ser muito menor que 10 semitons
    total_movement = sum(abs(f - c) for f, c in zip(sorted(f_voiced), sorted(c_voicing)))
    assert total_movement <= 4


# ---------------------------------------------------------------------------
# 4. Coerência das progressões
# ---------------------------------------------------------------------------
def test_progression_coherence():
    scale = Scale(root_note="C", scale_type=ScaleType.MAJOR)
    prog = ProgressionEngine(scale=scale, progression_tokens=["I", "vi", "IV", "V"], chord_duration_beats=4.0)

    # Ciclo total = 16 beats
    assert prog.total_beats == 16.0

    # Beat 0: I (C Major)
    chord0, idx0 = prog.get_chord_at_beat(0.0)
    assert chord0.root_note == "C"
    assert idx0 == 0

    # Beat 5 (compasso 2): vi (A Minor)
    chord1, idx1 = prog.get_chord_at_beat(5.0)
    assert chord1.root_note == "A"
    assert idx1 == 1

    # Sequência voiced gerada sem falhas
    voicings = prog.get_voiced_progression()
    assert len(voicings) == 4
    for v in voicings:
        assert len(v) >= 3


# ---------------------------------------------------------------------------
# 5. Estrutura das melodias
# ---------------------------------------------------------------------------
def test_melody_structure():
    scale = Scale(root_note="C", scale_type=ScaleType.MAJOR)
    prog = ProgressionEngine(scale=scale, progression_tokens=["I", "IV", "V", "I"])
    rhythm = RhythmEngine(bpm=60.0)
    melody_engine = MelodyEngine(scale=scale, progression=prog, rhythm=rhythm)

    rng = np.random.default_rng(1234)
    phrase = melody_engine.generate_phrase(
        start_beat=0.0,
        phrase_length_beats=8.0,
        rng=rng,
        density=0.6,
        prefer_chord_tones=True
    )

    assert len(phrase) > 0
    # Todas as notas devem pertencer à escala
    for note in phrase:
        assert scale.is_midi_in_scale(note.midi_note)
        assert note.duration_beats > 0
        assert 0.0 <= note.start_beat <= 8.0

    # Teste de arpejo
    c_chord = Chord(root_note="C", quality=ChordQuality.MAJOR, octave=4)
    arp = melody_engine.generate_arpeggio(chord=c_chord, start_beat=0.0, duration_beats=4.0, pattern="up")
    assert len(arp) == 8 # 4 beats / 0.5 step = 8 notas
    pitches = [n.midi_note for n in arp]
    # Notas do arpejo devem ser C, E, G
    for p in pitches:
        assert (p % 12) in [0, 4, 7]


# ---------------------------------------------------------------------------
# 6. Determinismo por seed e isolamento
# ---------------------------------------------------------------------------
def test_seed_determinism():
    ctx = RenderContext(duration_sec=5.0, seed=777)
    rm1 = RandomManager(master_seed=777)
    rm2 = RandomManager(master_seed=777)

    rng1 = rm1.get_plugin_rng("piano")
    rng2 = rm2.get_plugin_rng("piano")

    composer = Composer(context=ctx, root_note="D", scale_type="dorian")
    ctx.metadata["composer"] = composer

    piano1 = Piano()
    piano2 = Piano()

    piano1.prepare(ctx, rng1)
    piano2.prepare(ctx, rng2)

    chunk1 = piano1.render(0, 44100)
    chunk2 = piano2.render(0, 44100)

    np.testing.assert_array_equal(chunk1, chunk2)


def test_rng_isolation():
    rm = RandomManager(master_seed=999)
    rng_harp = rm.get_plugin_rng("harp")
    rng_flute = rm.get_plugin_rng("flute")

    vals_harp = rng_harp.uniform(0, 1, 50)
    vals_flute = rng_flute.uniform(0, 1, 50)

    assert not np.array_equal(vals_harp, vals_flute)


# ---------------------------------------------------------------------------
# 7. Continuidade de notas e envelopes entre chunks
# ---------------------------------------------------------------------------
def test_chunk_boundary_continuity():
    """
    Testa se uma nota longa que cruza a fronteira do chunk 1 (amostras 0..44100)
    e chunk 2 (amostras 44100..88200) é renderizada com continuidade matemática perfeita
    idêntica à renderização direta contínua.
    """
    from plugins.instruments.flute import FluteNoteEvent

    flute = Flute()
    flute.sample_rate = 44100
    # Nota que começa em 0.5s e termina em 1.5s (cruza a borda em 1.0s = amostra 44100)
    note = FluteNoteEvent(midi_note=69, start_sec=0.5, duration_sec=1.0, velocity=0.8)
    flute.add_note(note)

    # Renderização em chunks de 1 segundo (44100 amostras cada)
    chunk1 = flute.render(0, 44100)
    chunk2 = flute.render(44100, 44100)
    concatenated = np.concatenate((chunk1, chunk2), axis=0)

    # Renderização de uma só vez (88200 amostras)
    direct = flute.render(0, 88200)

    # Devem ser exatamente idênticos
    np.testing.assert_allclose(concatenated, direct, atol=1e-6)

    # Verifica se a junção em 44100 não tem salto/degrau (diferença finita suave)
    step_diff = np.abs(concatenated[44100] - concatenated[44099])
    assert np.max(step_diff) < 0.05 # Transição contínua sem clique


# ---------------------------------------------------------------------------
# 8. Validação individual dos nove instrumentos
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("plugin_class", ALL_INSTRUMENTS)
def test_all_nine_instruments(plugin_class):
    ctx = RenderContext(duration_sec=6.0, seed=42)
    composer = Composer(context=ctx, root_note="C", scale_type="lydian")
    ctx.metadata["composer"] = composer

    rm = RandomManager(master_seed=42)
    rng = rm.get_plugin_rng(plugin_class.PLUGIN_ID)

    inst = plugin_class()
    assert inst.category.value == "instrument"
    inst.prepare(ctx, rng)

    # Renderiza 2 segundos
    chunk = inst.render(0, 44100 * 2)

    assert chunk.shape == (88200, 2)
    assert chunk.dtype == np.float32
    assert not np.isnan(chunk).any()
    assert not np.isinf(chunk).any()
    # Verifica ausência de clipping grosseiro
    assert np.max(np.abs(chunk)) <= 1.5

    inst.reset()


# ---------------------------------------------------------------------------
# 9. Integração do Composer com o Core Engine
# ---------------------------------------------------------------------------
def test_composer_core_integration(tmp_path):
    output_wav = tmp_path / "music_integration.wav"
    ctx = RenderContext(
        sample_rate=44100,
        channels=2,
        chunk_duration_sec=2.0,
        duration_sec=6.0,
        seed=123,
        export_subtype="PCM_24"
    )

    composer = Composer(
        context=ctx,
        root_note="F",
        scale_type="lydian",
        bpm=60.0,
        a4_freq=432.0
    )
    ctx.metadata["composer"] = composer

    engine = Engine(context=ctx)

    # Registra e instancia múltiplos instrumentos
    for p_class in [AmbientPad, Bass, Harp, Bells]:
        engine.plugin_manager.register(p_class)
        engine.plugin_manager.create_instance(p_class.PLUGIN_ID)

    renderer = Renderer(engine)
    max_peak = renderer.render_to_file(output_wav)

    assert output_wav.exists()
    assert 0.05 < max_peak <= 1.0

    info = sf.info(str(output_wav))
    assert info.samplerate == 44100
    assert info.channels == 2
    assert info.subtype == "PCM_24"
    assert info.frames == 6 * 44100
