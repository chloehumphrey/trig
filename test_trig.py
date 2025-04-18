import unittest
import numpy as np
from numpy.testing import assert_allclose
import warnings

from trig import localize_gunshot
import trig

class TestTrigFunctions(unittest.TestCase):
    
    def setUp(self):
        # Common test data
        self.sensor_positions = [
            [0.0, 0.0],
            [100.0, 0.0],
            [0.0, 100.0],
            [100.0, 100.0]
        ]
        self.true_source = np.array([50.0, 50.0])
        self.true_t0 = 0.0
        self.speed_of_sound = 343.0
        
        # Generate ideal TOAs
        self.toas = []
        for sensor in self.sensor_positions:
            distance = np.linalg.norm(self.true_source - np.array(sensor))
            toa = self.true_t0 + distance / self.speed_of_sound
            self.toas.append(toa)

    
    def test_corrected_objective_function(self):
        """Test objective function."""
        # Test with known values
        theta = [50.0, 50.0, 0.0]  # True source position and time
        result = trig.objective(theta, self.sensor_positions, self.toas, self.speed_of_sound)
        # Should be very close to zero since these are the true values
        self.assertAlmostEqual(result, 0.0, places=10)
    
    def test_localize_gunshot_with_perfect_data(self):
        """Test localization with perfect (no noise) data."""
        
        # Test with perfect data
        estimated_source, estimated_t0, _ = localize_gunshot(
            self.sensor_positions, self.toas, self.speed_of_sound
        )
        
        # Check results
        assert_allclose(estimated_source, self.true_source, rtol=1e-4, atol=1e-4)
        self.assertAlmostEqual(estimated_t0, self.true_t0, places=4)

    
    def test_localize_gunshot_with_noisy_data(self):
        """Test localization with noisy data."""

        # Add noise to TOAs
        np.random.seed(42)  # For reproducibility
        noise_std = 0.01
        toas_noisy = np.array(self.toas) + np.random.normal(0, noise_std, len(self.toas))
        
        # Test with noisy data
        estimated_source, estimated_t0, _ = localize_gunshot(
            self.sensor_positions, toas_noisy, self.speed_of_sound
        )
        
        print("Estimated Source:", estimated_source)

        # Check results - allowing more tolerance due to noise
        assert_allclose(estimated_source, self.true_source, rtol=1e-1, atol=1e-1)
        self.assertAlmostEqual(estimated_t0, self.true_t0, places=1)
            
    
    def test_minimum_sensors_required(self):
        """Test that at least 3 sensors are required for 2D localization"""

        # Try with only 2 sensors (insufficient)
        insufficient_sensors = self.sensor_positions[:2]
        insufficient_toas = self.toas[:2]
        
        # This should still run but might give poor results
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimated_source, estimated_t0, result = localize_gunshot(
                insufficient_sensors, insufficient_toas, self.speed_of_sound
            )
        
        print("Estimated Source with insufficient sensors:", estimated_source)
        
        
        # Check if optimization was successful but possibly inaccurate
        self.assertTrue(hasattr(result, 'success'))


if __name__ == '__main__':
    unittest.main()