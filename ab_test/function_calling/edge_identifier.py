"""
AbsorptionEdgeIdentifier - Identify element and absorption edge from XAS spectra.

Identifies the element and edge type (K, L1, L2, L3) by:
  1. Loading a two-column (Energy, Mu) spectrum file
  2. Computing the first derivative d(mu)/d(E)
  3. Finding the energy at the maximum derivative (edge energy)
  4. Comparing against reference edge energies from the X-ray Data Booklet
"""

from __future__ import annotations

import os
from pathlib import Path


class EdgeIdentifier:
    def __init__(self):
        # Dictionary of K-edge energies (eV)
        self.k_edges = {
            "He": 24.6, "Li": 54.7, "Be": 111.5, "B": 188, "C": 284.2, "N": 409.9, "O": 543.1, "F": 696.7, "Ne": 870.2,
            "Na": 1070.8, "Mg": 1303, "Al": 1559, "Si": 1839, "P": 2145.5, "S": 2472, "Cl": 2822.4, "Ar": 3205.9,
            "K": 3608.4, "Ca": 4038.5, "Sc": 4492, "Ti": 4966, "V": 5465, "Cr": 5989, "Mn": 6539, "Fe": 7112,
            "Co": 7709, "Ni": 8333, "Cu": 8979, "Zn": 9659, "Ga": 10367, "Ge": 11103, "As": 11867, "Se": 12658,
            "Br": 13474, "Kr": 14326, "Rb": 15200, "Sr": 16105, "Y": 17038, "Zr": 17998, "Nb": 18986, "Mo": 20000,
            "Tc": 21044, "Ru": 22117, "Rh": 23220, "Pd": 24350, "Ag": 25514, "Cd": 26711, "In": 27940, "Sn": 29200,
            "Sb": 30491, "Te": 31814, "I": 33169, "Xe": 34561, "Cs": 35985, "Ba": 37441, "La": 38925, "Ce": 40443,
            "Pr": 41991, "Nd": 43569, "Pm": 45184, "Sm": 46834, "Eu": 48519, "Gd": 50239, "Tb": 51996, "Dy": 53789,
            "Ho": 55618, "Er": 57486, "Tm": 59390, "Yb": 61332, "Lu": 63314, "Hf": 65351, "Ta": 67416, "W": 69525,
            "Re": 71676, "Os": 73871, "Ir": 76111, "Pt": 78395, "Au": 80725, "Hg": 83102, "Tl": 85530, "Pb": 88005,
            "Bi": 90526, "Po": 93105, "At": 95730, "Rn": 98404, "Fr": 101137, "Ra": 103922, "Ac": 106755, "Th": 109651,
            "Pa": 112601, "U": 115606,
        }

        # Dictionary of L1-edge energies (eV)
        self.l1_edges = {
            "K": 378.6, "Ca": 438.4, "Sc": 498, "Ti": 564, "V": 626, "Cr": 696, "Mn": 769, "Fe": 844.6,
            "Co": 925.1, "Ni": 1008.6, "Cu": 1096.7, "Zn": 1196.2, "Ga": 1299, "Ge": 1414.6, "As": 1527,
            "Se": 1652, "Br": 1782, "Kr": 1921, "Rb": 2065, "Sr": 2216, "Y": 2373, "Zr": 2532, "Nb": 2698,
            "Mo": 2866, "Tc": 3043, "Ru": 3224, "Rh": 3412, "Pd": 3604, "Ag": 3806, "Cd": 4018, "In": 4238,
            "Sn": 4465, "Sb": 4698, "Te": 4939, "I": 5188, "Xe": 5453, "Cs": 5714, "Ba": 5989, "La": 6266,
            "Ce": 6549, "Pr": 6835, "Nd": 7126, "Pm": 7428, "Sm": 7737, "Eu": 8052, "Gd": 8376, "Tb": 8708,
            "Dy": 9046, "Ho": 9394, "Er": 9751, "Tm": 10116, "Yb": 10486, "Lu": 10870, "Hf": 11271, "Ta": 11682,
            "W": 12100, "Re": 12527, "Os": 12968, "Ir": 13419, "Pt": 13880, "Au": 14353, "Hg": 14839, "Tl": 15347,
            "Pb": 15861, "Bi": 16388, "Po": 16939, "At": 17493, "Rn": 18049, "Fr": 18639, "Ra": 19237,
            "Ac": 19840, "Th": 20472, "Pa": 21105, "U": 21757,
        }

        # Dictionary of L2-edge energies (eV)
        self.l2_edges = {
            "K": 297, "Ca": 350, "Sc": 403, "Ti": 461, "V": 521, "Cr": 584, "Mn": 649, "Fe": 720,
            "Co": 794, "Ni": 870, "Cu": 953, "Zn": 1044, "Ga": 1143, "Ge": 1248, "As": 1359,
            "Se": 1474, "Br": 1596, "Kr": 1731, "Rb": 1864, "Sr": 2007, "Y": 2156, "Zr": 2307, "Nb": 2465,
            "Mo": 2625, "Tc": 2793, "Ru": 2967, "Rh": 3146, "Pd": 3330, "Ag": 3524, "Cd": 3727, "In": 3938,
            "Sn": 4156, "Sb": 4380, "Te": 4612, "I": 4852, "Xe": 5107, "Cs": 5359, "Ba": 5624, "La": 5891,
            "Ce": 6164, "Pr": 6440, "Nd": 6722, "Pm": 7013, "Sm": 7312, "Eu": 7617, "Gd": 7930, "Tb": 8252,
            "Dy": 8581, "Ho": 8918, "Er": 9264, "Tm": 9617, "Yb": 9978, "Lu": 10349, "Hf": 10739, "Ta": 11136,
            "W": 11544, "Re": 11959, "Os": 12385, "Ir": 12824, "Pt": 13273, "Au": 13734, "Hg": 14209, "Tl": 14698,
            "Pb": 15200, "Bi": 15711, "Po": 16244, "At": 16785, "Rn": 17337, "Fr": 17907, "Ra": 18484,
            "Ac": 19083, "Th": 19693, "Pa": 20314, "U": 20948,
        }

        # Dictionary of L3-edge energies (eV)
        self.l3_edges = {
            "Sc": 402, "Ti": 456, "V": 513, "Cr": 575, "Mn": 640, "Fe": 708,
            "Co": 779, "Ni": 855, "Cu": 933, "Zn": 1020, "Ga": 1115, "Ge": 1217, "As": 1323, "Se": 1436,
            "Br": 1550, "Kr": 1675, "Rb": 1804, "Sr": 1941, "Y": 2080, "Zr": 2223, "Nb": 2371, "Mo": 2520,
            "Tc": 2677, "Ru": 2838, "Rh": 3004, "Pd": 3173, "Ag": 3351, "Cd": 3538, "In": 3730, "Sn": 3929,
            "Sb": 4132, "Te": 4341, "I": 4557, "Xe": 4782, "Cs": 5012, "Ba": 5247, "La": 5483, "Ce": 5723,
            "Pr": 5964, "Nd": 6208, "Pm": 6459, "Sm": 6716, "Eu": 6977, "Gd": 7243, "Tb": 7514, "Dy": 7790,
            "Ho": 8071, "Er": 8358, "Tm": 8648, "Yb": 8944, "Lu": 9244, "Hf": 9561, "Ta": 9881, "W": 10207,
            "Re": 10535, "Os": 10871, "Ir": 11215, "Pt": 11564, "Au": 11919, "Hg": 12284, "Tl": 12658, "Pb": 13035,
            "Bi": 13419, "Po": 13814, "At": 14214, "Rn": 14619, "Fr": 15031, "Ra": 15444, "Ac": 15871, "Th": 16300,
            "Pa": 16733, "U": 17166,
        }

    def load_spectrum(self, filepath):
        """
        Load a two-column (Energy, Mu) spectrum from a text file.

        Skips comment lines starting with '#' and blank lines.
        Expects whitespace- or tab-separated values.
        """
        energy = []
        mu = []
        try:
            with open(filepath, "r") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            energy.append(float(parts[0]))
                            mu.append(float(parts[1]))
                        except ValueError:
                            continue
            if not energy:
                return None, None
            return energy, mu
        except Exception as exc:
            print(f"Error loading {filepath}: {exc}")
            return None, None

    def find_edge_energy(self, energy, mu):
        """Find the edge energy by locating the maximum of the first derivative."""
        if len(energy) < 2:
            return None

        derivs = []
        mid_energies = []

        for i in range(len(energy) - 1):
            de = energy[i + 1] - energy[i]
            dmu = mu[i + 1] - mu[i]
            if de != 0:
                derivs.append(dmu / de)
                mid_energies.append((energy[i] + energy[i + 1]) / 2)
            else:
                derivs.append(0)
                mid_energies.append(energy[i])

        max_val = -float("inf")
        max_idx = -1
        for i, val in enumerate(derivs):
            if val > max_val:
                max_val = val
                max_idx = i

        if max_idx != -1:
            return mid_energies[max_idx]
        return None

    def identify_element(self, edge_energy, tolerance=100.0):
        """
        Identify the element and edge type based on the detected edge energy.

        Returns a sorted list of candidate matches, closest first.
        """
        if edge_energy is None:
            return []

        matches = []
        edge_priority = {"K": 0, "L3": 1, "L2": 2, "L1": 3}
        edge_tables = {
            "K": self.k_edges,
            "L1": self.l1_edges,
            "L2": self.l2_edges,
            "L3": self.l3_edges,
        }

        for edge_name, edge_dict in edge_tables.items():
            for element, energy in edge_dict.items():
                if abs(energy - edge_energy) <= tolerance:
                    matches.append(
                        {
                            "Element": element,
                            "Edge": edge_name,
                            "Energy": energy,
                            "Diff": abs(energy - edge_energy),
                        }
                    )

        matches.sort(key=lambda item: (item["Diff"], edge_priority.get(item["Edge"], 9)))
        return matches


def iter_cli_files(args: list[str]) -> list[str]:
    current_dir = Path(__file__).resolve().parents[1]
    test_data_dir = current_dir / "test_data"
    if args:
        return [args[0]]
    return [str(path) for path in sorted(test_data_dir.glob("*.txt"))]


def main(args: list[str] | None = None) -> int:
    identifier = EdgeIdentifier()

    for filepath in iter_cli_files(args or []):
        energy, mu = identifier.load_spectrum(filepath)
        edge_energy = identifier.find_edge_energy(energy, mu)
        matches = identifier.identify_element(edge_energy)
        if edge_energy is None or not matches:
            print(os.path.basename(filepath))
            print("  Could not identify an edge match.")
            print()
            continue

        print(f"{os.path.basename(filepath)}")
        print(f"  Edge energy: {edge_energy:.2f} eV")
        print(
            f"  Best match:  {matches[0]['Element']} {matches[0]['Edge']}-edge "
            f"(Diff: {matches[0]['Diff']:.2f} eV)"
        )
        print()

    return 0


__all__ = ["EdgeIdentifier", "iter_cli_files", "main"]
