from __future__ import absolute_import, print_function

import warnings

import numpy as np


class DeepFieldCalculator(object):
    """
    Thin wrapper around the ASE MACE calculator for qrefine.

    qrefine expects only a small backend contract:
      - run_qr(...)
      - set_label(...)
      - energy_free
      - forces
    """

    def __init__(self, device="cuda", enable_cueq=None):
        self.label = None
        self.method = None
        self.charge = 0
        self.energy_free = None
        self.forces = None
        self.calc = None
        self._enable_cueq_override = enable_cueq
        self.enable_cueq = False
        self._warned_pointcharges = False
        self.device = None
        self.set_device(device)

    def _create_calculator(self, model_path):
        from mace.calculators import MACECalculator

        return MACECalculator(
            model_paths=model_path,
            device=self.device,
            enable_cueq=self.enable_cueq,
        )

    def set_label(self, label):
        self.label = label

    def set_device(self, device):
        device = (device or "cuda").lower()
        if device not in ("cuda", "cpu"):
            raise ValueError("DeepField device must be one of: cuda, cpu")
        self.device = device
        if self._enable_cueq_override is None:
            self.enable_cueq = self.device == "cuda"
        else:
            self.enable_cueq = bool(self._enable_cueq_override)
        if self.method is not None:
            self.calc = self._create_calculator(self.method)

    def set_method(self, method):
        self.method = method
        self.calc = self._create_calculator(method)

    def set_charge(self, charge):
        self.charge = int(charge) if charge is not None else 0

    def run_qr(self, atoms, coordinates, charge, pointcharges, define_str=None):
        del coordinates, define_str
        if self.calc is None:
            raise RuntimeError(
                "DeepField calculator is not initialized. "
                "Set quantum.method to a valid DeepField/MACE model path."
            )

        active_charge = self.charge if charge is None else int(charge)
        atoms.info["charge"] = active_charge

        if pointcharges and not self._warned_pointcharges:
            warnings.warn(
                "DeepField/MACE point-charge embedding is not implemented in qrefine v1; "
                "pointcharges input will be ignored.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_pointcharges = True

        self.atoms = atoms
        atoms.calc = self.calc
        self.energy_free = float(atoms.get_potential_energy())
        self.forces = np.asarray(atoms.get_forces(), dtype=np.float64)
