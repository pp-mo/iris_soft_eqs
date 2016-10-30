import numpy as np

def co_nodata(coord):
    # Reduce a coord to a single dimension and single zero value, with possible
    # bounds, to enable metadata-only comparisons.
    # Also ignore dtypes.
    # NOTE: Surprisingly, standard coordinate equality *already* ignores dtype,
    # and also Coord subclass such as DimCord/AuxCoord.
    co_x = coord.copy()
    for _ in range(coord.ndim):
        co_x = co_x[0]
    co_x.points = [0]
    if co_x.has_bounds():
        co_x.bounds = [0, 0]
    return co_x


def corner_values(array):
    # Return a sorted list of the 'corner' values of an array.
    # I.E. all combinations of the first+last indices in each dimension,
    # resulting in an array of 2**ndim values.
    ndim = array.ndim
    for i_dim in range(ndim):
        slices = [slice(None)] * ndim
        slices[i_dim] = [0, -1]
        array = array[slices]
    return np.sort(array.flat)


def coord_endpoint_values(coord):
    # Return the min+max index point values of a coordinate, and add in the
    # bounds values if any.
    # N.B. multidimensional coords can't have bounds anyway.
    points = corner_values(coord.points)
    if coord.has_bounds():
        points = np.concatenate((points,
                                 corner_values(coord.bounds)))
    return points


def compare_coords(ref_coord, tst_coord):
    # A tolerant coordinate comparison check.
    # Allows for different ordering of dimensions, and possible inversion of
    # dimension directions (as in [..., ::-1, ...])
    if co_nodata(tst_coord) != co_nodata(ref_coord):
        msg = 'Coords {!r} have different metadata.'
        return False, msg.format(tst_coord.name())

    # Check the data-endpoint values (ignoring dimension directions).
    ref_ends = coord_endpoint_values(ref_coord)
    tst_ends = coord_endpoint_values(tst_coord)
    if not np.allclose(tst_ends, ref_ends):
        msg = 'Coords {!r} have significantly different values.'
        return False, msg.format(tst_coord.name())

    return True, ''


def compare_cubes(c1, c2):
    import numpy as np

    # Test sorted shapes, not actual, to allow flexible dimension ordering.
    if sorted(c1.shape) != sorted(c2.shape):
        return False, 'Cube shapes are incompatible.'

    if (c1.metadata != c2.metadata):
        return False, 'Cubes {!r} have different metadata.'.format(c1.name())

    #
    # Check they have essentially the same list of coordinates.
    #
    # get coords sorted by name.
    ref_names = sorted([c.name() for c in c1.coords()])
    tst_names = sorted([c.name() for c in c2.coords()])
    # Just check that all coords have different names, by which we can identify
    # them, the contrary is very undesirable and confusing !
    assert len(set(ref_names)) == len(ref_names)
    assert len(set(tst_names)) == len(tst_names)
    if tst_names != ref_names:
        # Don't have the 'same' coords overall : Try to explain the difference.
        msg = 'Cubes have different sets of coords: '
        missing = sorted(set(ref_names) - set(tst_names))
        extra = sorted(set(tst_names) - set(ref_names))
        if missing:
            msg += 'coords {} not found'.format(missing)
        if extra:
            if missing:
                msg += ' and '
            msg += 'additional coords {}'.format(extra)
        msg += ' in second "{}" cube.'.format(c1.name())
        return False, msg

    # There is only one common "set" of coord names.
    coord_names = ref_names

    # Check that the coord dimension mappings are equivalent.
    def dimension_coords_map(cube):
        # Return a dict of {dim: sorted tuple of coords mapped to dim}.
        per_dim_coords = {dim:set() for dim in range(cube.ndim)}
        for coord in cube.coords():
            for dim in cube.coord_dims(coord):
                # Include in the set of coords mapping to this dimension.
                per_dim_coords[dim].add(coord.name())
        # Return a dict with all the values recast as frozen sets, so we can
        # more easily make sets of those.
        return {dim:frozenset(coords_set)
                for dim, coords_set in per_dim_coords.items()}

    ref_dim_coords = dimension_coords_map(c1)
    tst_dim_coords = dimension_coords_map(c2)
    ref_coord_dim_groups = set(coords for coords in ref_dim_coords.values())
    tst_coord_dim_groups = set(coords for coords in tst_dim_coords.values())
    if ref_coord_dim_groups != tst_coord_dim_groups:
        # Note: this allows dimensions to appear in a different order in the
        # cube, or in a multidimensional coordinate.
        return False, 'Cubes have incompatible dimension mappings.'

    # Finally compare the coords themselves, but allowing for possible
    # different dimension orderings and inverted dimension directions.
    for name in coord_names:
        ref_coord, tst_coord = (cube.coord(name) for cube in (c1, c2))
        match, msg = compare_coords(ref_coord, tst_coord)
        if not match:
            return False, msg

    return True, ''


def compare_cubelists(cl1, cl2):
    if len(cl1) != len(cl2):
        return False, 'cubelists of different lengths'
    set1 = set(cl1)
    set2 = set(cl2)
    result_pairs = []
    for c1 in set1:
        found = False
        for c2 in list(set2):
            found, _ = compare_cubes(c1, c2)
            if found:
                set2.remove(c2)
                result_pairs.append((c1, c2))
                break
        if not found:
            msg = 'cube#1:\n{}\n\n.. not found in ..\n\n{}'
            return False, msg.format(c1, cl2)
    assert set2 == set()
    return True, ''