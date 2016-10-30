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


if __name__ == '__main__':
    tst1()
    tst2()

