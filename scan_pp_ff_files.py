from glob import glob
import os
import os.path

from iris.experimental.fieldsfile import load as structured_load
from iris import load as normal_load

from soft_iris_compares import compare_cubes, compare_cubelists


datazoo_path = '/data/local/dataZoo'


def all_files_iter(base_path):
    for dirpath, dirnames, filenames in os.walk(base_path, followlinks=True):
        for filename in filenames:
            yield os.path.join(dirpath, filename)

def all_ff_files_iter():
    for filepath in all_files_iter(os.path.join(datazoo_path, 'FF')):
        yield filepath

def all_pp_files_iter(non_pp_only=False):
    for filepath in all_files_iter(os.path.join(datazoo_path, 'PP')):
        is_pp = filepath.endswith('.pp')
        if ((non_pp_only and not is_pp) or
            (not non_pp_only and is_pp)):
                yield filepath

def all_pp_dirs():
    base_path = os.path.join(datazoo_path, 'PP')
    return (dirpath
            for dirpath, dirnames, filenames
            in os.walk(base_path, followlinks=True)
            if any(filename.endswith('.pp') for filename in filenames))


def sample_pp_files(n_max_per_dir=3):
    for pp_dirpath in all_pp_dirs():
        filenames = glob(os.path.join(pp_dirpath, '*.pp'))
        if n_max_per_dir:
            filenames = filenames[:n_max_per_dir]
        filenames = [os.path.join(pp_dirpath, filename)
                     for filename in filenames]
        for i_name in range(1, len(filenames)):
            filenames[i_name] = '  + ' + filenames[i_name]
        for filename in filenames:
            yield filename

def show_all_files():
    print
    print 'FF files...'
    print '\n'.join('{!r},'.format(path)
                    for path in all_ff_files_iter())
    
    print
    print 'PP files...'
    print '\n'.join('{!r},'.format(path)
                    for path in all_pp_files_iter(non_pp_only=False))


def tst1():
    ccs = structured_load('/data/local/dataZoo/PP/mogrepssubsets/2010031612.qteg12.oper18.060.pp468.pp')
    ccn = normal_load('/data/local/dataZoo/PP/mogrepssubsets/2010031612.qteg12.oper18.060.pp468.pp')
    tsts = ccs[3]
    tstn = ccn[5]
    result, msg = compare_cubes(tsts, tstn)
    print 'Cubes equal? = ', result
    print msg

def tst2():
    # Compare all results from loading file twice...
    ccs = structured_load('/data/local/dataZoo/PP/mogrepssubsets/2010031612.qteg12.oper18.060.pp468.pp')
    ccn = normal_load('/data/local/dataZoo/PP/mogrepssubsets/2010031612.qteg12.oper18.060.pp468.pp')
    result, msg = compare_cubelists(ccs, ccn)
    print 'Cubelists equal? = ', result
    print msg

    #
    # NOTE:
    #  this fails to match the 'plain' (i.e. not min or max) air_temp cubes
    #  because the pressure coords have opposite direction orders:
    #  'normal' has decreasing values 1000...250, 'structured' is ascending
    #

def tst_compare_all_files(filenames):
#    for filename in sample_pp_files(1):
#    for filename in all_ff_files_iter():
    for filename in filenames:
        skip = False
        truename = filename
        if filename.startswith('#'):
            truename = filename[1:].strip()
            skip = True
        megs = os.stat(truename).st_size * 1.0e-6
        if megs > 350.0:
            skip = True

        print
        print '{}   {:8.3f}Mb'.format(filename.ljust(60), megs)
        if skip:
            print '  ((skip))'
            continue

        try:
            d_normal = normal_load(filename)
        except Exception as err:
            msg = '  XXX normal load fails : {}'.format(err)
        else:
            try:
                d_struct = structured_load(filename)
            except Exception as err:
                msg = '  --- structured load fails : ' + str(err)
            else:
                result, err_msg = compare_cubelists(d_normal, d_struct)
                if result:
                    assert err_msg == ''
                    msg = '  + OK'
                else:
                    msg = '  -- MATCH FAIL: ' + err_msg
        print msg

def tst_compare_pps():
    with open('selected_pp_files.txt') as fo:
        filenames = fo.readlines()
    filenames = [filename.strip()
                 for filename in filenames]
    filenames = [filename for filename in filenames if len(filename) > 0]
    tst_compare_all_files(filenames)

def tst_compare_ffs():
    tst_compare_all_files(all_ff_files_iter())


if __name__ == '__main__':
#    tst1()
#    tst2()
#    print '\n'.join(sample_pp_files(4))
    tst_compare_pps()
