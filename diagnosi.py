# Lancia con il python del .venv (quello dove c'e' ixpeobssim):
#   .\.venv\Scripts\python.exe diagnosi.py     (oppure semplicemente: python diagnosi.py con (.venv) attivo)
import numpy as np
import astropy.io.fits as pf

PATH = '/Users/gabri/OneDrive/Fisica/TESI/QPO_python/QPO_swift/Obs1/'
DU = [f'{PATH}ixpe02250901_det{i}_evt2_v01_src.fits' for i in (1, 2, 3)]

try:
    from ixpeobssim.evt.event import xEventFile
except Exception as e:
    xEventFile = None
    print('ixpeobssim non importabile in questo python:', e)

du_id = np.load(PATH + 'du_id_final.npy')
en_fin = np.load(PATH + 'energy_final.npy')

for i, f in enumerate(DU, start=1):
    hdu = pf.open(f)
    ev = hdu['EVENTS'].data
    print(f'\n--- DU{i} ---')
    print('colonne EVENTS:', ev.columns.names)
    mm = np.load(PATH + f'mask_{i}_allDU.npy').astype(bool)
    e_fin = en_fin[du_id == i]
    print('eventi grezzi:', len(ev), '| dopo mask merging:', mm.sum(), '| DU in times_final:', len(e_fin))
    if mm.sum() != len(e_fin):
        print('ATTENZIONE: conteggi diversi, ordine non verificabile')
        continue

    # energia "ixpeobssim" (quella che usa xpselect)
    e_ixpe = None
    if xEventFile is not None:
        ef = xEventFile(f)
        print('attributi con "energ" in xEventFile:', [m for m in dir(ef) if 'energ' in m.lower()])
        for name in [m for m in dir(ef) if 'energ' in m.lower() and not m.startswith('__')]:
            try:
                cand = getattr(ef, name)
                if callable(cand):
                    cand = cand()
                cand = np.asarray(cand)
            except Exception:
                continue
            if cand.ndim == 1 and len(cand) == len(ev) and np.issubdtype(cand.dtype, np.number):
                e_ixpe = cand[mm]
                print(f'energia ixpeobssim da ef.{name}')
                break
        if e_ixpe is None:
            print('nessun attributo array di energia trovato in xEventFile')

    e_pi = ev['PI'][mm] * 0.04
    print('range energy_final:', e_fin.min(), e_fin.max())
    print('max |energy_final - PI*0.04|    :', np.abs(e_fin - e_pi).max())
    n_fin = ((e_fin >= 2.0) & (e_fin <= 3.5)).sum()
    print('eventi 2.0-3.5 con energy_final :', n_fin)
    print('eventi 2.0-3.5 con PI*0.04      :', ((e_pi >= 2.0) & (e_pi <= 3.5)).sum())
    if e_ixpe is not None:
        print('max |energy_final - ixpeobssim| :', np.abs(e_fin - e_ixpe).max())
        print('eventi 2.0-3.5 con ixpeobssim   :', ((e_ixpe >= 2.0) & (e_ixpe <= 3.5)).sum())
import numpy as np
P='/Users/gabri/OneDrive/Fisica/TESI/QPO_python/QPO_swift/Obs1/'
d, e = np.load(P+'du_id_final.npy'), np.load(P+'energy_final.npy')
print([int((np.isclose(e[d==i], 3.5)).sum()) for i in (1,2,3)])