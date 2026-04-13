import pytest
import os
import tempfile
import numpy as np
import astropy.units as u
from astropy.time import Time

from cosipy.phase_resolved_analysis.ephemeris import Ephemeris
from cosipy.phase_resolved_analysis.phase_selector import PhaseSelector, apply_phase_exposure_correction

def test_ephemeris_phase_calculation():
    """Test that the constant frequency phase calculation is correct."""
    t0 = Time('2024-01-01T00:00:00', scale='utc')
    f0 = 30.0 * u.Hz
    ephem = Ephemeris(f0, t0)
    
    # 0.05 seconds later at 30Hz = 1.5 rotations -> Phase should be 0.5
    test_time = t0 + 0.05 * u.s
    phase = ephem.get_phase(test_time)
    np.testing.assert_allclose(phase, 0.5, atol=1e-5)

def test_ephemeris_duty_cycle():
    """Test that the duty cycle calculates the correct physical time fraction."""
    t0 = Time('2024-01-01T00:00:00', scale='utc')
    ephem = Ephemeris(30.0 * u.Hz, t0)
    
    t_start = t0
    t_stop = t0 + 10 * u.s
    intervals = [(0.1, 0.2), (0.8, 0.9)] # Total width = 0.2 (20%)
    
    duty_cycle = ephem.get_duty_cycle(t_start, t_stop, intervals)
    
    # 20% of 10 seconds is 2.0 seconds
    assert duty_cycle.value == pytest.approx(2.0)
    assert duty_cycle.unit == u.s

def test_ephemeris_from_par_file():
    """Test parsing F0 from a standard TEMPO .par file."""
    # Create a temporary mock .par file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        # Using Fortran 'D' notation to ensure our parser handles it
        f.write("PSRJ J0534+2200\nF0 29.6D0\n")
        temp_name = f.name
        
    try:
        t0 = Time('2024-01-01T00:00:00', scale='utc')
        ephem = Ephemeris.from_par_file(temp_name, t0)
        assert ephem.f0.to_value(u.Hz) == 29.6
    finally:
        os.remove(temp_name)

def test_phase_selector_validation():
    """Test that PhaseSelector rejects invalid interval inputs."""
    t0 = Time('2024-01-01T00:00:00', scale='utc')
    ephem = Ephemeris(30.0 * u.Hz, t0)
    
    # Valid
    selector = PhaseSelector(ephem, [(0.1, 0.3)])
    assert selector.intervals == [(0.1, 0.3)]
    
    # Invalid bounds (> 1.0)
    with pytest.raises(ValueError):
        PhaseSelector(ephem, [(0.8, 1.2)])
        
    # Invalid order (start > stop)
    with pytest.raises(ValueError):
        PhaseSelector(ephem, [(0.8, 0.2)])


class MockLivetimeHist:
    def __init__(self, contents):
        self.contents = contents

class MockSpacecraftHistory:
    """A lightweight mock to test the exposure utility without loading actual FITS data."""
    def __init__(self, tstart, tstop, livetime_array):
        self.intervals_tstart = tstart
        self.intervals_tstop = tstop
        self._livetime_hist = MockLivetimeHist(livetime_array)

def test_apply_phase_exposure_correction():
    """Test that the utility correctly scales the histpy contents array."""
    t0 = Time('2024-01-01T00:00:00', scale='utc')
    ephem = Ephemeris(30.0 * u.Hz, t0)
    
    # Mock 2 orientation bins of 10 seconds each
    tstart = Time(['2024-01-01T00:00:00', '2024-01-01T00:00:10'], scale='utc')
    tstop = Time(['2024-01-01T00:00:10', '2024-01-01T00:00:20'], scale='utc')
    
    # Assume the detector was on for 8.0s in the first bin, and 9.0s in the second
    livetime = np.array([8.0, 9.0]) * u.s 
    
    sc_hist = MockSpacecraftHistory(tstart, tstop, livetime)
    
    # Apply a 20% phase cut
    intervals = [(0.1, 0.3)]
    corrected_sc = apply_phase_exposure_correction(sc_hist, ephem, intervals)
    
    # The new livetime should be exactly 20% of the original livetime per bin
    np.testing.assert_allclose(corrected_sc._livetime_hist.contents.value, [1.6, 1.8])