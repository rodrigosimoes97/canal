"""
Testes de reprodutibilidade e determinismo do RandomManager.
"""
import numpy as np
from core.random_manager import RandomManager


def test_random_manager_reproducibility():
    rm1 = RandomManager(master_seed=42)
    rm2 = RandomManager(master_seed=42)

    rng1 = rm1.get_plugin_rng("pad_synth")
    rng2 = rm2.get_plugin_rng("pad_synth")

    vals1 = rng1.uniform(0, 1, 100)
    vals2 = rng2.uniform(0, 1, 100)

    np.testing.assert_array_equal(vals1, vals2)


def test_random_manager_seed_variation():
    rm1 = RandomManager(master_seed=42)
    rm2 = RandomManager(master_seed=999)

    rng1 = rm1.get_plugin_rng("pad_synth")
    rng2 = rm2.get_plugin_rng("pad_synth")

    vals1 = rng1.uniform(0, 1, 100)
    vals2 = rng2.uniform(0, 1, 100)

    assert not np.array_equal(vals1, vals2)


def test_random_manager_plugin_isolation():
    rm = RandomManager(master_seed=42)

    rng_a = rm.get_plugin_rng("plugin_a")
    rng_b = rm.get_plugin_rng("plugin_b")

    # Amostrar do plugin A não pode interferir no plugin B
    vals_a1 = rng_a.normal(0, 1, 50)
    vals_b1 = rng_b.normal(0, 1, 50)

    assert not np.array_equal(vals_a1, vals_b1)

    # Nova instância: se pedirmos B primeiro, B deve produzir os mesmos valores exatos
    rm_fresh = RandomManager(master_seed=42)
    rng_b_fresh = rm_fresh.get_plugin_rng("plugin_b")
    vals_b_fresh = rng_b_fresh.normal(0, 1, 50)

    np.testing.assert_array_equal(vals_b1, vals_b_fresh)


def test_random_manager_time_event_determinism():
    rm = RandomManager(master_seed=42)

    # Dois geradores criados para o mesmo timestamp do mesmo plugin
    rng_t1 = rm.get_time_event_rng("bells", 120)
    rng_t2 = rm.get_time_event_rng("bells", 120)

    # Devem produzir a mesma saída
    assert rng_t1.integers(0, 100) == rng_t2.integers(0, 100)

    # Timestamp diferente deve produzir saída distinta
    rng_t3 = rm.get_time_event_rng("bells", 121)
    # Muito improvável colisão
    assert rng_t1.uniform() != rng_t3.uniform()
