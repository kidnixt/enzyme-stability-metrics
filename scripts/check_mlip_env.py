"""Smoke test del entorno MLIP. Corre esto ANTES de escribir nada de X3.

    python scripts/check_mlip_env.py

Verifica, en orden: torch ve la GPU, la capacidad de computo soporta lo que
vamos a usar, y UMA carga y devuelve una energia sobre una molecula chica.
"""
import sys

def main():
    import torch
    print(f"torch {torch.__version__}  cuda disponible: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("!! sin GPU visible; UMA va a correr en CPU y sera lentisimo")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name}  compute capability {cap[0]}.{cap[1]}  {mem:.1f} GB")
    if cap[0] < 8:
        print("!! arquitectura previa a Ampere: NO hay bfloat16 nativo.")
        print("   Si algo intenta bf16 va a ser lento o fallar. Usar fp32.")

    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    from ase.build import molecule

    # uma-s-1 (sin sufijo) tiene un bug conocido de extensividad y NO sirve
    # para energias de interaccion, que son exactamente lo que calculamos.
    CHECKPOINT = "uma-s-1p1"
    print(f"cargando {CHECKPOINT} ...")
    predictor = pretrained_mlip.get_predict_unit(CHECKPOINT, device="cuda")
    calc = FAIRChemCalculator(predictor, task_name="omol")

    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1})   # neutro singlete, explicito
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    print(f"H2O  E = {e:.6f} eV   |F|max = {abs(f).max():.6f} eV/A")
    print("OK: el entorno MLIP funciona.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
