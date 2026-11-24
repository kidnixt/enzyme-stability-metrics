"""Lo que UMA sabe por dentro: sacar las representaciones atomicas aprendidas.

La idea
-------
Hasta ahora de UMA usamos la salida: una energia, y de ahi cinco escalares por
molecula. Pero para producir esa energia el modelo construye una
representacion interna de CADA ATOMO, que codifica su entorno tridimensional
completo: que atomos tiene cerca, a que distancia, en que geometria. Eso son
cientos de dimensiones por atomo que nunca tocamos.

Si esa representacion captura quimica util, agrupada por molecula deberia
funcionar como una huella digital aprendida y consciente de la geometria 3D -
algo que los descriptores clasicos, que son casi todos bidimensionales, no
tienen.

Esto NO es medir una falla mas. Es construir una representacion y ver si gana.
Puede salir positivo, que es lo que ninguno de los experimentos anteriores
podia.

Como se saca
------------
Sin depender de que fairchem exponga una API para esto: se enganchan hooks de
PyTorch en los modulos internos. Un hook ve el tensor que sale de cada capa
durante la pasada hacia adelante, exista o no una funcion publica que lo
devuelva.

Por eso el script corre en dos fases:

    --inspect   imprime el arbol de modulos y la forma del tensor que produce
                cada uno sobre una molecula de prueba. Sirve para elegir de
                donde sacar la representacion, en vez de adivinar.

    (default)   engancha el modulo elegido, corre las 761 moleculas y escribe
                la huella de cada una.

Uso:
    python scripts/uma_embeddings.py --inspect
    python scripts/uma_embeddings.py --layer <nombre que haya salido>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", default="data/processed/molecules.sdf")
    ap.add_argument("--out", default="data/processed/mol_embeddings.csv")
    ap.add_argument("--model", default="uma-s-1p2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--inspect", action="store_true",
                    help="solo mirar que hay adentro, no extraer nada")
    ap.add_argument("--layer", default="",
                    help="nombre exacto del modulo a enganchar")
    ap.add_argument("--max-depth", type=int, default=3,
                    help="profundidad del arbol de modulos al inspeccionar")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from ase import Atoms
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    from rdkit import Chem

    predictor = pretrained_mlip.get_predict_unit(args.model, device=args.device)
    calc = FAIRChemCalculator(predictor, task_name="omol")

    # el modulo de torch esta en algun lado del predictor; se busca sin suponer
    net = None
    for attr in ("model", "module", "_model", "backbone"):
        cand = getattr(predictor, attr, None)
        if isinstance(cand, torch.nn.Module):
            net = cand
            print(f"modulo torch encontrado en predictor.{attr}")
            break
    if net is None:
        for name, val in vars(predictor).items():
            if isinstance(val, torch.nn.Module):
                net = val
                print(f"modulo torch encontrado en predictor.{name}")
                break
    if net is None:
        raise SystemExit("no se encontro el nn.Module adentro del predictor; "
                         "corre --inspect y pegame la salida")

    mols = [m for m in Chem.SDMolSupplier(args.sdf, removeHs=False) if m is not None]
    if args.limit:
        mols = mols[: args.limit]

    def to_atoms(m):
        conf = m.GetConformer()
        pos = np.array([list(conf.GetAtomPosition(i))
                        for i in range(m.GetNumAtoms())])
        a = Atoms(symbols=[x.GetSymbol() for x in m.GetAtoms()], positions=pos)
        n_elec = int(m.GetProp("n_elec"))
        a.info.update({"charge": int(m.GetProp("charge")),
                       "spin": 1 if n_elec % 2 == 0 else 2})
        a.calc = calc
        return a

    # ---------------- fase 1: mirar que hay
    if args.inspect or not args.layer:
        print("\n" + "=" * 72)
        print("Modulos internos y forma del tensor que produce cada uno")
        print("=" * 72)
        print("Se busca algo con forma (n_atomos, D): una fila por atomo.\n")

        seen = {}
        hooks = []

        def mk(name):
            def hook(mod, inp, out):
                t = out
                if isinstance(t, (tuple, list)) and t:
                    t = t[0]
                if isinstance(t, dict):
                    t = next((v for v in t.values()
                              if isinstance(v, torch.Tensor)), None)
                if isinstance(t, torch.Tensor):
                    seen[name] = tuple(t.shape)
            return hook

        for name, mod in net.named_modules():
            if not name or name.count(".") >= args.max_depth:
                continue
            hooks.append(mod.register_forward_hook(mk(name)))

        probe = to_atoms(mols[0])
        n_at = len(probe)
        try:
            probe.get_potential_energy()
        finally:
            for h in hooks:
                h.remove()

        print(f"molecula de prueba: {mols[0].GetProp('smiles')}  "
              f"({n_at} atomos)\n")
        hits = [(n, s) for n, s in seen.items() if s and s[0] == n_at and len(s) == 2]
        for name, shape in sorted(seen.items()):
            mark = "   <- UNA FILA POR ATOMO" if (name, shape) in hits else ""
            print(f"   {name:52s} {str(shape):>18s}{mark}")
        print(f"\n{len(hits)} candidatos con una fila por atomo.")
        if hits:
            print("El de mas adelante en la red suele ser el mas informativo:")
            for n, s in hits[-3:]:
                print(f"   --layer {n}    (D = {s[1]})")
        print("\nElegi uno y corre de nuevo con --layer <nombre>.")
        return

    # ---------------- fase 2: extraer
    target = dict(net.named_modules()).get(args.layer)
    if target is None:
        raise SystemExit(f"no existe el modulo '{args.layer}'; corre --inspect")

    buf = {}

    def hook(mod, inp, out):
        t = out
        if isinstance(t, (tuple, list)) and t:
            t = t[0]
        buf["x"] = t.detach().float().cpu().numpy()

    h = target.register_forward_hook(hook)
    rows, n_err = [], 0
    print(f"extrayendo de '{args.layer}' sobre {len(mols)} moleculas\n")

    try:
        for k, m in enumerate(mols):
            try:
                a = to_atoms(m)
                a.get_potential_energy()
                x = buf.get("x")
                if x is None or x.ndim != 2 or x.shape[0] != len(a):
                    n_err += 1
                    continue
            except Exception as e:
                n_err += 1
                print(f"   {m.GetProp('mol_id')}: {type(e).__name__}")
                continue
            # tres agrupamientos: el promedio describe la molecula entera,
            # el maximo destaca el atomo mas extremo en cada dimension
            feats = np.concatenate([x.mean(0), x.max(0), x.min(0)])
            rows.append({"mol_id": m.GetProp("mol_id"),
                         "y": int(m.GetProp("y")),
                         "split": m.GetProp("split"),
                         **{f"emb{i}": float(v) for i, v in enumerate(feats)}})
            if (k + 1) % 50 == 0:
                print(f"   {k+1}/{len(mols)}")
    finally:
        h.remove()

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nescrito {args.out}: {len(df)} moleculas x "
          f"{df.shape[1]-3} dimensiones   (fallos: {n_err})")


if __name__ == "__main__":
    main()
