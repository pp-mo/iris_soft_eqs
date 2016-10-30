from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import iris.tests as tests

import numpy as np

from iris.cube import Cube, CubeList
from iris.coords import DimCoord, AuxCoord

from soft_iris_compares import (compare_coords,
                                compare_cubes,
                                compare_cubelists)


def _check_compare_result(testcase, func, item1, item2, fail_msg=''):
    # Exercise a 'compare' function
    result, msg = func(item1, item2)
    if not fail_msg:
        testcase.assertEqual(msg, '')
        testcase.assertTrue(result)
    else:
        testcase.assertIn(fail_msg, msg)
        testcase.assertFalse(result)


class TestCoordsMetadata(tests.IrisTest):
    def setUp(self):
        self.ref_co_1d = DimCoord([1, 2])
        self.co_a = DimCoord([1, 2], long_name='a')
        self.msg_metadata = "Coords 'a' have different metadata"

    def _coords_eq(self, c1, c2, fail_msg=''):
        _check_compare_result(self, compare_coords, c1, c2, fail_msg=fail_msg)

    def test_self_eq(self):
        c1 = self.ref_co_1d
        self._coords_eq(c1, c1)

    def test_dup(self):
        c1 = self.ref_co_1d
        c2 = DimCoord([1, 2])
        self._coords_eq(c1, c2)

    def test_dtype_change(self):
        c1 = self.ref_co_1d
        c2 = DimCoord(np.array([1.0, 2.0], dtype=np.float64))
        self._coords_eq(c1, c2)

    def test_coordtype_change(self):
        c1 = self.ref_co_1d
        c2 = AuxCoord([1, 2])
        self._coords_eq(c1, c2)

    def test_fail_attributes_extra(self):
        c1 = self.co_a
        c2 = c1.copy()
        c2.attributes['this'] = 'that'
        self._coords_eq(c1, c2, fail_msg=self.msg_metadata)

    def test_fail_attributes_missing(self):
        c1 = self.co_a
        c1.attributes['this'] = 'that'
        c1.attributes['that'] = 'other'
        c2 = c1.copy()
        del c2.attributes['this']
        self._coords_eq(c1, c2, fail_msg=self.msg_metadata)

    def test_fail_attributes_differ(self):
        c1 = self.co_a
        c1.attributes['this'] = 'that'
        c2 = c1.copy()
        c2.attributes['this'] = 'other'
        self._coords_eq(c1, c2, fail_msg=self.msg_metadata)

    def test_fail_units_differ(self):
        c1 = self.co_a
        c2 = c1.copy()
        c2.units = 'm'
        self._coords_eq(c1, c2, fail_msg=self.msg_metadata)

    def test_fail_long_names_differ(self):
        c1 = self.co_a
        c1.rename('air_temperature')
        c2 = c1.copy()
        c2.long_name = 'x'
        msg = ("Coords 'air_temperature' have "
               "different metadata")
        self._coords_eq(c1, c2, fail_msg=msg)

    def test_fail_varnames_differ(self):
        c1 = self.co_a
        c2 = c1.copy()
        c2.var_name = 'q'
        self._coords_eq(c1, c2, fail_msg=self.msg_metadata)

    #
    # NOTE: coord_system is also a possible metadata mismatch.
    # ?? should we ignore that one ??
    # probably not, leave it in.
    #


class TestCoordsValues(tests.IrisTest):
    def setUp(self):
        self.ref_1d = DimCoord([1, 2, 3, 4], long_name='a')
        self.ref_1d_bounded = self.ref_1d.copy()
        self.ref_1d_bounded.guess_bounds()
        self.ref_multidim = AuxCoord(
            [[[1, 2, 3], [6, 5, 4]],
             [[0, 0, 1], [9, 9, 9]]],
            long_name='b')
        self.value_msg = "significantly different values"

    def _coords_eq(self, c1, c2, fail_msg=''):
        _check_compare_result(self, compare_coords, c1, c2, fail_msg=fail_msg)

    def test_1d_invert(self):
        c1 = self.ref_1d
        c2 = c1.copy()
        c2 = c2[::-1]
        self._coords_eq(c1, c2)

    def test_fail_1d_points_differ(self):
        c1 = self.ref_1d
        c2 = c1.copy()
        pts = c2.points.copy()
        pts[-1] += 5
        c2.points = pts
        self._coords_eq(c1, c2, fail_msg=self.value_msg)

    def test_fail_1d_bounds_differ(self):
        c1 = self.ref_1d_bounded
        c2 = c1.copy()
        bds = c2.bounds.copy()
        bds[0, 0] -= 5.0
        c2.bounds = bds
        self._coords_eq(c1, c2, fail_msg=self.value_msg)

    def test_multidim_invert_0(self):
        c1 = self.ref_multidim
        c2 = c1.copy()
        pts = c2.points.copy()
        pts = pts[::-1]
        c2.points = pts
        self._coords_eq(c1, c2)

    def test_multidim_invert_1(self):
        c1 = self.ref_multidim
        c2 = c1.copy()
        pts = c2.points.copy()
        pts = pts[:, ::-1, :]
        c2.points = pts
        self._coords_eq(c1, c2)

    def test_multidim_invert_02(self):
        c1 = self.ref_multidim
        c2 = c1.copy()
        pts = c2.points.copy()
        pts = pts[::-1, :, ::-1]
        c2.points = pts
        self._coords_eq(c1, c2)


class TestCubesMetadata(tests.IrisTest):
    def setUp(self):
        self.ref_cube_a = Cube([1, 2, 3], long_name='a')
        self.msg_metadata = "Cubes 'a' have different metadata"

    def _cubes_eq(self, c1, c2, fail_msg=''):
        _check_compare_result(self, compare_cubes, c1, c2, fail_msg=fail_msg)

    def test_self_eq(self):
        c1 = self.ref_cube_a
        self._cubes_eq(c1, c1)

    def test_fail_dtype_change(self):
        c1 = self.ref_cube_a
        c2 = c1.copy()
        c2.data = np.array(c1.data, dtype=np.float64)
        self._cubes_eq(c1, c2)

    def test_fail_attributes_extra(self):
        c1 = self.ref_cube_a
        c2 = c1.copy()
        c2.attributes['this'] = 'that'
        self._cubes_eq(c1, c2, fail_msg=self.msg_metadata)

    def test_fail_attributes_missing(self):
        c1 = self.ref_cube_a
        c1.attributes['this'] = 'that'
        c1.attributes['that'] = 'other'
        c2 = c1.copy()
        del c2.attributes['this']
        self._cubes_eq(c1, c2, fail_msg=self.msg_metadata)

    def test_fail_attributes_differ(self):
        c1 = self.ref_cube_a
        c1.attributes['this'] = 'that'
        c2 = c1.copy()
        c2.attributes['this'] = 'other'
        self._cubes_eq(c1, c2, fail_msg=self.msg_metadata)

    def test_fail_units_differ(self):
        c1 = self.ref_cube_a
        c1.units = 'm'
        c2 = c1.copy()
        c2.convert_units('ft')
        self._cubes_eq(c1, c2, fail_msg=self.msg_metadata)

    def test_fail_long_names_differ(self):
        c1 = self.ref_cube_a
        c1.rename('air_temperature')
        c2 = c1.copy()
        c2.long_name = 'x'
        msg = ("Cubes 'air_temperature' have "
               "different metadata")
        self._cubes_eq(c1, c2, fail_msg=msg)

    def test_fail_varnames_differ(self):
        c1 = self.ref_cube_a
        c2 = c1.copy()
        c2.var_name = 'q'
        self._cubes_eq(c1, c2, fail_msg=self.msg_metadata)


class TestCubesCoordLists(tests.IrisTest):
    def setUp(self):
        self.cube_a = Cube([[1, 2, 3], [4, 5, 6]], long_name='a')
        self.x_coord = DimCoord([11, 12, 13], long_name='x')
        self.y_coord = DimCoord([21, 22], long_name='y')
        self.cube_a.add_dim_coord(self.x_coord, 1)
        self.cube_a.add_dim_coord(self.y_coord, 0)
        self.msg_metadata = "Cubes 'a' have different metadata"

    def _cubes_eq(self, c1, c2, fail_msg=''):
        _check_compare_result(self, compare_cubes, c1, c2, fail_msg=fail_msg)

    def test_self_eq(self):
        c1 = self.cube_a
        self._cubes_eq(c1, c1)

    def test_fail_coords_missing(self):
        c1 =self.cube_a
        c2 = c1.copy()
        c2.remove_coord('y')
        msg = ("Cubes have different sets of coords: "
               "coords ['y'] not found in second \"a\" cube")
        self._cubes_eq(c1, c2, fail_msg=msg)

    def test_fail_coords_extra(self):
        c1 =self.cube_a
        c2 = c1.copy()
        c1.remove_coord('y')
        msg = ("Cubes have different sets of coords: "
               "additional coords ['y'] in second \"a\" cube")
        self._cubes_eq(c1, c2, fail_msg=msg)

    def test_fail_coords_differ(self):
        c1 =self.cube_a
        c2 = c1.copy()
        c1.remove_coord('x')
        c2.remove_coord('y')
        msg = ("Cubes have different sets of coords: "
               "coords ['y'] not found "
               "and additional coords ['x'] in second \"a\" cube")
        self._cubes_eq(c1, c2, fail_msg=msg)


class TestCubesDimsAndCoords(tests.IrisTest):
    def setUp(self):
        self.cube_a = Cube([[1, 2, 3], [4, 5, 6]], long_name='a')
        self.x_coord = DimCoord([11, 12, 13], long_name='x')
        self.y_coord = DimCoord([21, 22], long_name='y')
        self.cube_a.add_dim_coord(self.x_coord, 1)
        self.cube_a.add_dim_coord(self.y_coord, 0)

    def _cubes_eq(self, c1, c2, fail_msg=''):
        _check_compare_result(self, compare_cubes, c1, c2, fail_msg=fail_msg)

    def test_transpose(self):
        c1 = self.cube_a
        c2 = c1.copy()
        c2.transpose((1, 0))
        self._cubes_eq(c1, c2)

    def test_invert(self):
        c1 = self.cube_a
        c2 = c1[:, ::-1]
        self._cubes_eq(c1, c2)

    def test_fail_shapes_differ(self):
        c1 = self.cube_a
        c2 = c1[:-1]
        self._cubes_eq(c1, c2, fail_msg='Cube shapes are incompatible')

    def test_multidim_dims_differ(self):
        c1 = self.cube_a
        c2 = c1.copy()
        data2d = np.arange(6).reshape(2, 3)
        co_2x3 = AuxCoord(data2d, long_name='twod')
        c1.add_aux_coord(co_2x3, (0, 1))
        co_3x2 = AuxCoord(data2d.transpose(), long_name='twod')
        c2.add_aux_coord(co_3x2, (1, 0))
        self._cubes_eq(c1, c2)

    def test_fail_coords_differ(self):
        c1 =self.cube_a
        c2 = c1.copy()
        c2.coord('x').attributes['extra'] = 1
        self._cubes_eq(c1, c2, fail_msg="Coords 'x' have different metadata")


if __name__ == '__main__':
    import sys
    sys.argv.append('-v')
    tests.main()
