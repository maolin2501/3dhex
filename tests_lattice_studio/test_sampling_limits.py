"""Exercise large COUNT estimates only: never allocate the proposed grids."""
import unittest
from unittest.mock import patch
from latticefill.studio_jobs import (validate_design, estimate, physical_memory,
                                    PREVIEW_VOXEL_LIMIT, FINAL_VOXEL_LIMIT)


class SamplingLimitTests(unittest.TestCase):
    def setUp(self):
        self.config = validate_design(dict(lattice='kelvin', cell=4, wall=.6,
                                          pitch=.2, axis='z', scope='body', quality='final'))
        self.bounds = [[0,0,0], [128.5700579466,128.5998285523,149.2130979419]]

    @patch('latticefill.studio_jobs.physical_memory', return_value=dict(total_gb=32, available_gb=12))
    def test_previous_320_million_rejection_is_now_allowed_with_warning(self, _):
        result = estimate(self.bounds, self.config)
        self.assertEqual(result['limit'], 400_000_000)
        self.assertGreater(result['voxels'], 319_000_000)
        self.assertLess(result['voxels'], 321_000_000)
        self.assertTrue(result['allowed'])
        self.assertEqual(result['pitch'], .2)  # final mode must not coarsen
        self.assertTrue(result['memory_risk'])
        self.assertEqual(result['memory_mode'],'chunked')
        self.assertGreater(result['memory_gb'],12)  # mesh post-processing included
        self.assertLess(result['memory_gb'],16)
        self.assertGreater(result['scratch_gb'],2)
        self.assertAlmostEqual(result['legacy_memory_gb'],19.2,places=1)

    @patch('latticefill.studio_jobs.physical_memory', return_value=None)
    def test_preview_is_capped_at_32_million(self, _):
        result = estimate(self.bounds, self.config | {'quality':'preview'})
        self.assertEqual(result['limit'], 32_000_000)
        self.assertGreater(result['voxels'], 8_000_000)
        self.assertLessEqual(result['voxels'], PREVIEW_VOXEL_LIMIT)
        self.assertTrue(result['allowed_by_count'])
        self.assertFalse(result['allowed'])
        self.assertFalse(result['feature_resolved'])
        self.assertGreater(result['pitch'], .2)
        self.assertIsNone(result['physical_memory'])
        self.assertTrue(any('无法读取' in s for s in result['warnings']))

    @patch('latticefill.studio_jobs.physical_memory', return_value=dict(total_gb=128, available_gb=96))
    def test_new_cap_still_rejects_oversized_final_grid(self, _):
        result = estimate([[0,0,0],[300,300,300]], self.config)
        self.assertGreater(result['voxels'], FINAL_VOXEL_LIMIT)
        self.assertFalse(result['allowed'])
        self.assertEqual(result['pitch'], .2)

    @patch('latticefill.studio_jobs.physical_memory', return_value=dict(total_gb=128, available_gb=96))
    def test_more_available_ram_removes_memory_warning_not_count_limit(self, _):
        result = estimate(self.bounds, self.config)
        self.assertTrue(result['allowed'])
        self.assertFalse(result['memory_risk'])
        self.assertEqual(result['limit'], FINAL_VOXEL_LIMIT)

    def test_real_memory_snapshot_is_optional_and_sensible(self):
        ram = physical_memory()
        if ram is not None:
            self.assertGreater(ram['total_gb'], 0)
            self.assertGreaterEqual(ram['available_gb'], 0)
            self.assertLessEqual(ram['available_gb'], ram['total_gb'])
