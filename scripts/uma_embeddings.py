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

La trampa de la equivarianza
----------------------------
UMA es una red equivariante: la representacion de cada atomo no es un vector
plano sino (S, D), donde S son componentes esfericos. Para L=2 son 9: uno de
l=0, tres de l=1, cinco de l=2. Los de l>0 ROTAN CON LA MOLECULA.

Usarlos crudos como features seria un error silencioso: girar la misma
molecula cambiaria la prediccion, lo cual no tiene sentido para clasificar.
Asi que se construyen invariantes - el componente l=0, que es escalar, y la
norma de cada bloque l>0 - y despues se verifica empiricamente rotando una
molecula, en vez de confiar en que el razonamiento de arriba sea correcto.

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

    --check-invariance
                extrae la huella de una molecula, la rota al azar, la vuelve a
                extraer y compara. Si no coinciden, las features estan rotas y
                no hay que seguir.

    (default)   engancha el modulo elegido, corre las 761 moleculas y escribe
                la huella de cada una.

Uso:
    python scripts/uma_embeddings.py --inspect
    python scripts/uma_embeddings.py --layer module.backbone --check-invariance
    python scripts/uma_embeddings.py --layer module.backbone
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
    ap.add_argument("--check-invariance", action="store_true")
    ap.add_argument("--pool", default="mean,max",
                    help="agrupamiento sobre atomos: mean, max, min")
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
        hits = [(n, s) for n, s in seen.items()
                if s and s[0] == n_at and len(s) in (2, 3)]
        for name, shape in sorted(seen.items()):
            mark = ""
            if (name, shape) in hits:
                mark = ("   <- POR ATOMO, EQUIVARIANTE" if len(shape) == 3
                        else "   <- POR ATOMO, plano")
            print(f"   {name:52s} {str(shape):>18s}{mark}")
        print(f"\n{len(hits)} candidatos con una fila por atomo.")
        if hits:
            print("El de mas adelante en la red suele ser el mas informativo:")
            for n, s in hits[-4:]:
                d = f"S = {s[1]}, D = {s[2]}" if len(s) == 3 else f"D = {s[1]}"
                print(f"   --layer {n}    ({d})")
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
        if isinstance(t, torch.Tensor):
            buf["x"] = t.detach().float().cpu().numpy()

    def invariants(x):
        """De (n_atomos, S, D) o (n_atomos, D) a (n_atomos, K) invariante.

        Para S = (L+1)^2 componentes esfericos, el bloque l ocupa las
        posiciones l^2 .. (l+1)^2. El bloque l=0 es un escalar, invariante tal
        cual. Los bloques l>0 rotan, pero su NORMA no: por cada l>0 se guarda
        la norma del bloque, que son D numeros mas.
        """
        if x.ndim == 2:
            return x
        n, S, D = x.shape
        L = int(round(S ** 0.5)) - 1
        if (L + 1) ** 2 != S:
            # no es un cuadrado perfecto: se usa la norma sobre S, conservador
            return np.linalg.norm(x, axis=1)
        parts = [x[:, 0, :]]                       # l = 0, ya invariante
        for l in range(1, L + 1):
            blk = x[:, l * l:(l + 1) * (l + 1), :]
            parts.append(np.linalg.norm(blk, axis=1))
        return np.concatenate(parts, axis=1)

    def fingerprint(x, pools):
        inv = invariants(x)
        out = []
        for p_ in pools:
            out.append({"mean": inv.mean(0), "max": inv.max(0),
                        "min": inv.min(0)}[p_])
        return np.concatenate(out)

    pools = [p_.strip() for p_ in args.pool.split(",") if p_.strip()]
    h = target.register_forward_hook(hook)

    # ---- chequeo de invarianza: rotar la molecula no debe cambiar la huella
    if args.check_invariance:
        try:
            m = mols[0]
            a = to_atoms(m)
            a.get_potential_energy()
            f1 = fingerprint(buf["x"], pools)
            raw1 = buf["x"].copy()

            rng = np.random.default_rng(0)
            q = rng.normal(size=(3, 3))
            R, _ = np.linalg.qr(q)
            if np.linalg.det(R) < 0:
                R[:, 0] *= -1                      # rotacion propia, sin reflejar
            b = to_atoms(m)
            b.set_positions(b.get_positions() @ R.T)
            b.calc = calc
            b.get_potential_energy()
            f2 = fingerprint(buf["x"], pools)
            raw2 = buf["x"]
        finally:
            h.remove()

        d_inv = float(np.abs(f1 - f2).max())
        esc = float(max(np.abs(f1).max(), 1e-9))
        d_raw = float(np.abs(raw1 - raw2).max())
        print("=" * 68)
        print("Chequeo: la huella sobrevive a rotar la molecula?")
        print("=" * 68)
        print(f"   molecula          : {mols[0].GetProp('smiles')}")
        print(f"   dimensiones       : {len(f1)}")
        print(f"   cambio en el tensor CRUDO al rotar : {d_raw:.4f}")
        print(f"   cambio en la HUELLA al rotar       : {d_inv:.2e}")
        print(f"   relativo a su escala               : {d_inv/esc:.2e}")
        print()
        if d_inv / esc < 1e-3:
            print("   OK: la huella es invariante. El crudo si cambia, que es")
            print("   justamente la razon por la que no se puede usar crudo.")
        else:
            print("   ROTO: la huella cambia al rotar. No seguir: cualquier")
            print("   modelo entrenado sobre esto aprenderia la orientacion")
            print("   arbitraria que le dio RDKit al generar la geometria.")
        return

    rows, n_err = [], 0
    print(f"extrayendo de '{args.layer}' sobre {len(mols)} moleculas")
    print(f"agrupamiento: {', '.join(pools)}\n")

    try:
        for k, m in enumerate(mols):
            try:
                a = to_atoms(m)
                a.get_potential_energy()
                x = buf.get("x")
                if x is None or x.shape[0] != len(a):
                    n_err += 1
                    continue
                feats = fingerprint(x, pools)
            except Exception as e:
                n_err += 1
                print(f"   {m.GetProp('mol_id')}: {type(e).__name__}")
                continue
            rows.append({"mol_id": m.GetProp("mol_id"),
                         "y": int(m.GetProp("y")),
                         "split": m.GetProp("split"),
                         **{f"emb{i}": float(v) for i, v in enumerate(feats)}})
            if (k + 1) % 100 == 0:
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
