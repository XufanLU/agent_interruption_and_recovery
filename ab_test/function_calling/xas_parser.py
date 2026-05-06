"""Flexible parser for the raw XAS text and CSV formats used in this project.

This module contains the format-detection and column-extraction logic that
turns heterogeneous beamline/export files into normalized energy/mu arrays for
the rest of the XAS workflow stack.
"""

import os
import math
import re
import sys
import shlex

class XASParser:
    """
    Flexible parser for various XAS data formats.
    
    Supported formats:
      - Standard XDI (# Column.N: label, space-separated data)
      - CLS/BioXAS multi-event (# column N: label, comma-separated data, dual event headers)
      - LabVIEW/APS 9-BM (# Here is a readable list of column headings, space-separated data)
      - Generic header-line (# energy  mu  ..., space/tab data)
      - CSV with header (energy, It, IpreKB, fluo_DT_corrected, etc.)
      - Simple XY (2-column text: Energy Mu)
    """

    def __init__(self):
        pass

    def parse_file(self, filepath):
        """
        Backward-compatible wrapper that returns only the sample spectrum.
        """
        energy, mu, _ = self.parse_file_with_metadata(filepath)
        return energy, mu

    def parse_file_with_metadata(self, filepath):
        """
        Parses a file to extract Energy, sample Mu, and optional auxiliary spectra.
        Returns (energy_list, mu_list, metadata) or (None, None, {}) if failed.
        """
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None, None, {}

        try:
            with open(filepath, 'r', errors='ignore') as f:
                lines = f.readlines()
        except:
            return None, None, {}

        if not lines:
            return None, None, {}

        # Try parsers in order of specificity
        
        # 1. CLS multi-event format (comma-separated, dual event columns)
        energy, mu, metadata = self._parse_cls_format(lines)
        if energy and len(energy) > 5:
            return energy, mu, metadata

        # 2. LabVIEW/APS 9-BM format (# Here is a readable list of column headings)
        energy, mu, metadata = self._parse_labview_9bm(lines)
        if energy and len(energy) > 5:
            return energy, mu, metadata

        # 3. Standard XDI column-defined (# Column.N: label)
        energy, mu, metadata = self._parse_xdi_columns(lines)
        if energy and len(energy) > 5:
            return energy, mu, metadata
            
        # 4. Header-line parser (# energy  i0  itrans ...)
        energy, mu, metadata = self._parse_header_line(lines)
        if energy and len(energy) > 5:
            return energy, mu, metadata

        # 5. CSV with header (energy, It, IpreKB, fluo_DT_corrected, ...)
        energy, mu, metadata = self._parse_csv_header(lines)
        if energy and len(energy) > 5:
            return energy, mu, metadata

        # 6. Simple XY fallback
        energy, mu, metadata = self._parse_simple_xy(lines)
        if energy and len(energy) > 5:
            return energy, mu, metadata

        print(f"Could not parse {filepath}")
        return None, None, {}

    # ─── Column label classification ─────────────────────────────────

    def _classify_column(self, label):
        """Classify a column label into a semantic role."""
        label = label.lower().strip()
        
        # Energy column
        if any(kw in label for kw in ['energy', 'mono']):
            # Exclude setpoints and feedback variants if we have a plain energy
            if 'fbk' in label or 'feedback' in label:
                return 'energy_fbk'
            if 'setpoint' in label or ':sp' in label:
                return 'energy_setpoint'
            if 'setting' in label:
                return 'energy_setting'
            return 'energy'
        
        # Mu / normalized
        if label in ('mu', 'xmu', 'mu(e)', 'mutrans', 'mufluor'):
            return 'mu'
        if 'norm' in label:
            return 'mu'
        
        # I0 detector
        if label in ('i0', 'i0detector', 'i0detector_darkcorrect', 'iprekb', 'ipreslit'):
            return 'i0'
        if 'i0' in label and 'i10' not in label:
            return 'i0'
        if 'prekb' in label or 'preslit' in label:
            return 'i0'
        
        # Transmission detector (I1, IT, or itrans)
        if label in ('i1', 'it', 'itrans', 'itransmission', 'i1detector', 'i1detector_darkcorrect'):
            return 'itrans'
        if 'itrans' in label:
            return 'itrans'
        if label == 'i1' or (label.startswith('i1') and 'i10' not in label and 'i11' not in label):
            return 'itrans'
        
        # Fluorescence detector
        if any(kw in label for kw in ['fluor', 'fluo', 'pips', 'mca', 'idiode', 'if', 'ifluor']):
            return 'ifluor'
        
        # Reference detector
        if any(kw in label for kw in ['irefer', 'iref', 'i2']):
            return 'irefer'
            
        return 'unknown'

    def _find_data_start(self, lines, start_from=0):
        """Find the first line that looks like numeric data."""
        for i in range(start_from, len(lines)):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                continue
            # Try parsing as numbers
            try:
                # Handle comma-separated
                if ',' in line:
                    parts = line.split(',')
                else:
                    parts = line.split()
                if len(parts) >= 2:
                    float(parts[0])
                    float(parts[1])
                    return i
            except (ValueError, IndexError):
                continue
        return -1

    def _compute_mu(self, parts, col_roles):
        """Compute mu from a data row given column role assignments."""
        # If we have a direct mu column, use it
        if 'mu' in col_roles:
            idx = col_roles['mu']
            if idx < len(parts):
                return float(parts[idx])
        
        # Otherwise compute from detectors
        i0_val = None
        it_val = None
        if_val = None
        
        if 'i0' in col_roles and col_roles['i0'] < len(parts):
            i0_val = float(parts[col_roles['i0']])
        if 'itrans' in col_roles and col_roles['itrans'] < len(parts):
            it_val = float(parts[col_roles['itrans']])
        if 'ifluor' in col_roles and col_roles['ifluor'] < len(parts):
            if_val = float(parts[col_roles['ifluor']])
        
        if i0_val is not None and i0_val != 0:
            if it_val is not None:
                # Transmission: mu = -ln(It/I0)
                ratio = it_val / i0_val
                if ratio > 0:
                    return -math.log(ratio)
            if if_val is not None:
                # Fluorescence: mu = If/I0  
                return if_val / i0_val
        
        return None

    def _compute_reference_mu(self, parts, col_roles):
        """
        Compute an embedded foil/reference spectrum from It and Iref when both
        channels are present in the same scan.
        """
        it_val = None
        iref_val = None

        if 'itrans' in col_roles and col_roles['itrans'] < len(parts):
            it_val = float(parts[col_roles['itrans']])
        if 'irefer' in col_roles and col_roles['irefer'] < len(parts):
            iref_val = float(parts[col_roles['irefer']])

        if it_val is not None and iref_val is not None and it_val > 0 and iref_val > 0:
            ratio = iref_val / it_val
            if ratio > 0:
                return -math.log(ratio)

        return None

    def _split_parts(self, line, delimiter):
        if delimiter == 'csv':
            return [p.strip() for p in line.split(',')]
        return line.split()

    def _collect_spectra(self, lines, data_start, energy_col, col_roles, delimiter):
        energies = []
        mus = []
        reference_energies = []
        reference_mus = []

        for line in lines[data_start:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = self._split_parts(line, delimiter)
            try:
                if energy_col >= len(parts):
                    continue
                e_val = float(parts[energy_col])
                mu_val = self._compute_mu(parts, col_roles)
                ref_mu_val = self._compute_reference_mu(parts, col_roles)
                if mu_val is not None:
                    energies.append(e_val)
                    mus.append(mu_val)
                if ref_mu_val is not None:
                    reference_energies.append(e_val)
                    reference_mus.append(ref_mu_val)
            except (ValueError, IndexError):
                continue

        metadata = {}
        if len(reference_energies) > 5 and len(reference_energies) == len(reference_mus):
            metadata["reference_energy"] = reference_energies
            metadata["reference_mu"] = reference_mus

        return energies, mus, metadata

    # ─── CLS multi-event format ──────────────────────────────────────

    def _parse_cls_format(self, lines):
        """
        Parse CLS/BioXAS format with multi-event column definitions.
        
        Two sub-formats:
        A) '# column N: label' definitions (e.g., TMAO Arsenic file)
        B) '#(1) Event-ID "EnergySetting" "EnergyFeedback" ... "I0" "I1" ...'
           where human-readable labels are in quotes on the #(1) line.
        
        Data is always comma-separated.
        """
        # Detect CLS format
        is_cls = False
        for line in lines:
            if 'CLS Data Acquisition' in line:
                is_cls = True
                break
        
        if not is_cls:
            return None, None, {}
        
        # Try sub-format A: "# column N: label" with Event markers
        col_defs = self._parse_cls_column_defs(lines)
        
        # Try sub-format B: "#(1) ... quoted-labels" if A didn't work
        if not col_defs:
            col_defs = self._parse_cls_parenthetical_defs(lines)
        
        if not col_defs:
            return None, None, {}
        
        # Classify columns
        col_roles = {}
        energy_col = -1
        
        for col_num, label in col_defs.items():
            idx = col_num - 1  # Convert to 0-indexed
            role = self._classify_column(label)
            
            if role in ('energy', 'energy_fbk', 'energy_setpoint', 'energy_setting'):
                if 'energy' not in col_roles:
                    col_roles['energy'] = idx
                    energy_col = idx
            elif role == 'i0' and 'i0' not in col_roles:
                col_roles['i0'] = idx
            elif role == 'itrans' and 'itrans' not in col_roles:
                col_roles['itrans'] = idx
            elif role == 'ifluor' and 'ifluor' not in col_roles:
                col_roles['ifluor'] = idx
            elif role == 'mu' and 'mu' not in col_roles:
                col_roles['mu'] = idx
            elif role == 'irefer' and 'irefer' not in col_roles:
                col_roles['irefer'] = idx
        
        if energy_col == -1:
            return None, None, {}
        
        has_mu = 'mu' in col_roles
        has_detectors = 'i0' in col_roles and ('itrans' in col_roles or 'ifluor' in col_roles)
        if not has_mu and not has_detectors:
            return None, None, {}
        
        # Find data start
        data_start = self._find_data_start(lines)
        if data_start == -1:
            return None, None, {}

        return self._collect_spectra(lines, data_start, energy_col, col_roles, 'csv')

    def _parse_cls_column_defs(self, lines):
        """Parse CLS '# column N: label' definitions (sub-format A)."""
        col_defs = {}  # col_num (1-indexed) -> label
        in_event1 = False
        
        for line in lines:
            stripped = line.strip().lower()
            if '# event:' in stripped and 'readmcs' in stripped:
                in_event1 = True
                continue
            if ('# event:' in stripped and 'background' in stripped) or \
               ('# id: 2' in stripped and in_event1):
                in_event1 = False
                continue
            
            if in_event1 and stripped.startswith('#') and 'column' in stripped:
                try:
                    content = stripped.lstrip('#').strip()
                    content = content.replace('column', '').strip()
                    colon_idx = content.find(':')
                    if colon_idx != -1:
                        num_str = content[:colon_idx].strip()
                        label = content[colon_idx+1:].strip()
                        if num_str.isdigit():
                            col_defs[int(num_str)] = label
                    else:
                        parts = content.split(None, 1)
                        if len(parts) >= 2 and parts[0].isdigit():
                            col_defs[int(parts[0])] = parts[1]
                except:
                    pass
        
        return col_defs

    def _parse_cls_parenthetical_defs(self, lines):
        """
        Parse CLS '#(1)' format (sub-format B).
        
        These files have two sets of #(1) lines:
          1. PV names:   #(1) Event-ID BL1606-ID-1:Energy MONO:Energy:fbk ...
          2. Labels:     #(1) Event-ID "Energy Setting" "$(EnergyFeedback)" ... "I0" "I1" ...
        
        We use the human-readable label line (the one with quotes).
        """
        col_defs = {}
        
        # Find #(1) lines with quoted labels
        label_line = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#(1)') and '"' in stripped:
                label_line = stripped
                break
        
        if not label_line:
            # Try without quotes - use the PV-based #(1) line
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('#(1)') and ':' in stripped:
                    label_line = stripped
                    break
        
        if not label_line:
            return col_defs
        
        # Parse the label line
        content = label_line[4:].strip()  # Remove '#(1)'
        
        # Try shlex to handle quotes
        try:
            labels = shlex.split(content, comments=False)
        except:
            labels = content.split()
        
        # Map to 1-indexed column numbers
        for i, label in enumerate(labels):
            col_defs[i + 1] = label.lower()
        
        return col_defs

    # ─── LabVIEW/APS 9-BM format ─────────────────────────────────────

    def _parse_labview_9bm(self, lines):
        """
        Parse LabVIEW Control Panel / APS Beamline 9-BM format.
        
        Uses "Here is a readable list of column headings" to get column mapping:
          #  1) Mono Energy *        11) XMAP8:0:ZnKa         
          #  5) I0                   15) XMAP8:4:ZnKa         
          #  6) IT                   16) XMAP8:5:ZnKa         
          #  8) IF                   ...
        
        Data is space-separated and starts after "# Column Headings:" line.
        """
        if not any('Here is a readable list of column headings' in line for line in lines):
            return None, None, {}
        
        col_defs = {}  # 1-indexed col_num -> label
        
        for line in lines:
            stripped = line.strip()
            if 'Here is a readable list of column headings' in stripped:
                # Start parsing from this section
                pass
            if not stripped.startswith('#') or ')' not in stripped:
                continue
            # Match " 1) Mono Energy *" or " 11) XMAP8:0:ZnKa"
            for m in re.finditer(r'(\d+)\)\s+([^\d][^)]*?)(?=\s+\d+\)|$)', stripped):
                num_str, label = m.group(1), m.group(2).strip()
                if num_str.isdigit():
                    col_defs[int(num_str)] = label
        
        if not col_defs:
            return None, None, {}
        
        # Classify columns
        col_roles = {}
        energy_col = -1
        for col_num, label in col_defs.items():
            idx = col_num - 1
            role = self._classify_column(label)
            if role in ('energy', 'energy_fbk', 'energy_setpoint', 'energy_setting'):
                if 'energy' not in col_roles:
                    col_roles['energy'] = idx
                    energy_col = idx
            elif role == 'i0' and 'i0' not in col_roles:
                col_roles['i0'] = idx
            elif role == 'itrans' and 'itrans' not in col_roles:
                col_roles['itrans'] = idx
            elif role == 'ifluor' and 'ifluor' not in col_roles:
                col_roles['ifluor'] = idx
            elif role == 'mu' and 'mu' not in col_roles:
                col_roles['mu'] = idx
            elif role == 'irefer' and 'irefer' not in col_roles:
                col_roles['irefer'] = idx
        
        if energy_col == -1:
            return None, None, {}
        
        has_mu = 'mu' in col_roles
        has_detectors = 'i0' in col_roles and ('itrans' in col_roles or 'ifluor' in col_roles)
        if not has_mu and not has_detectors:
            return None, None, {}
        
        # Find data start (first numeric line after "# Column Headings:")
        data_start = -1
        for i, line in enumerate(lines):
            if 'Column Headings:' in line and line.strip().startswith('#'):
                for j in range(i + 1, len(lines)):
                    ln = lines[j].strip()
                    if not ln or ln.startswith('#'):
                        continue
                    # Verify it looks like data (starts with a number)
                    parts = ln.split()
                    if len(parts) >= 2:
                        try:
                            float(parts[0])
                            float(parts[1])
                            data_start = j
                            break
                        except ValueError:
                            continue
                break
        if data_start == -1:
            data_start = self._find_data_start(lines)
        if data_start == -1:
            return None, None, {}

        return self._collect_spectra(lines, data_start, energy_col, col_roles, 'space')

    # ─── Standard XDI column format ──────────────────────────────────

    def _parse_xdi_columns(self, lines):
        """
        Parse standard XDI format with # Column.N: label definitions.
        Data is space-separated.
        """
        col_defs = {}  # col_num (1-indexed) -> label
        
        for line in lines:
            stripped = line.strip().lower()
            if not stripped.startswith('#'):
                continue
            
            # Match: # Column.1: energy eV   or   # column 1: energy
            content = stripped.lstrip('#').strip()
            
            # Try "Column.N:" format
            if content.startswith('column'):
                content_after = content[6:].strip()  # after "column"
                # Could be ".1:" or " 1:"
                content_after = content_after.lstrip('.')
                # Split on ':' or space
                colon_idx = content_after.find(':')
                if colon_idx != -1:
                    num_str = content_after[:colon_idx].strip()
                    label = content_after[colon_idx+1:].strip()
                else:
                    parts = content_after.split(None, 1)
                    if len(parts) >= 2:
                        num_str = parts[0]
                        label = parts[1]
                    else:
                        continue
                
                num_str = num_str.rstrip(':').strip()
                if num_str.isdigit():
                    col_defs[int(num_str)] = label
        
        if not col_defs:
            return None, None, {}
        
        # Classify columns
        col_roles = {}
        energy_col = -1
        
        for col_num, label in col_defs.items():
            idx = col_num - 1  # 0-indexed
            role = self._classify_column(label)
            
            if role in ('energy', 'energy_fbk', 'energy_setpoint', 'energy_setting'):
                if 'energy' not in col_roles:
                    col_roles['energy'] = idx
                    energy_col = idx
            elif role == 'i0' and 'i0' not in col_roles:
                col_roles['i0'] = idx
            elif role == 'itrans' and 'itrans' not in col_roles:
                col_roles['itrans'] = idx
            elif role == 'ifluor' and 'ifluor' not in col_roles:
                col_roles['ifluor'] = idx
            elif role == 'mu' and 'mu' not in col_roles:
                col_roles['mu'] = idx
            elif role == 'irefer':
                col_roles['irefer'] = idx
        
        if energy_col == -1:
            return None, None, {}
        
        has_mu = 'mu' in col_roles
        has_detectors = 'i0' in col_roles and ('itrans' in col_roles or 'ifluor' in col_roles)
        if not has_mu and not has_detectors:
            return None, None, {}

        # Find data start
        data_start = self._find_data_start(lines)
        if data_start == -1:
            return None, None, {}

        return self._collect_spectra(lines, data_start, energy_col, col_roles, 'space')

    # ─── Header-line parser ──────────────────────────────────────────

    def _parse_header_line(self, lines):
        """
        Parse files with a column-name header line like:
          #  energy  i0  itrans  irefer
        or:
          # Energy, Mu, Normalized:
        """
        header_line = ""
        data_start_idx = 0
        potential_headers = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('#'):
                clean = stripped.lstrip('#').strip()
                # Heuristic: must contain "energy" (case-insensitive) and have multiple words/columns
                low = clean.lower()
                if 'energy' in low and (len(clean.split()) > 1 or ',' in clean):
                    potential_headers.append((i, clean))
            elif not stripped:
                continue
            else:
                # Reached first data line
                if potential_headers:
                    last_idx, last_content = potential_headers[-1]
                    header_line = last_content
                    data_start_idx = i
                break
        
        if not header_line:
            return None, None, {}
        
        # Parse header columns
        # Remove trailing colons (e.g. "Energy, Mu, Normalized:")
        header_line = header_line.rstrip(':')
        
        if ',' in header_line:
            cols = [c.strip() for c in header_line.split(',')]
        else:
            try:
                cols = shlex.split(header_line, comments=False)
            except:
                cols = header_line.split()
        
        if len(cols) < 2:
            cols = header_line.replace('\t', ' ').split()
        
        # Classify columns
        col_roles = {}
        energy_col = -1
        
        for i, col_name in enumerate(cols):
            role = self._classify_column(col_name)
            if role in ('energy', 'energy_fbk', 'energy_setpoint', 'energy_setting'):
                if 'energy' not in col_roles:
                    col_roles['energy'] = i
                    energy_col = i
            elif role == 'i0' and 'i0' not in col_roles:
                col_roles['i0'] = i
            elif role == 'itrans' and 'itrans' not in col_roles:
                col_roles['itrans'] = i
            elif role == 'ifluor' and 'ifluor' not in col_roles:
                col_roles['ifluor'] = i
            elif role == 'mu' and 'mu' not in col_roles:
                col_roles['mu'] = i
            elif role == 'irefer' and 'irefer' not in col_roles:
                col_roles['irefer'] = i
        
        if energy_col == -1:
            return None, None, {}
        
        # If we only have energy and no other roles, assume col 1 is mu
        has_mu = 'mu' in col_roles
        has_detectors = 'i0' in col_roles and ('itrans' in col_roles or 'ifluor' in col_roles)
        if not has_mu and not has_detectors:
            # Fallback: if energy is col 0, treat col 1 as mu
            if energy_col == 0 and len(cols) >= 2:
                col_roles['mu'] = 1
            else:
                return None, None, {}

        delimiter = 'csv' if ',' in header_line else 'space'
        return self._collect_spectra(lines, data_start_idx, energy_col, col_roles, delimiter)

    # ─── CSV with header ───────────────────────────────────────────

    def _parse_csv_header(self, lines):
        """
        Parse CSV files where the first line is a comma-separated header.
        
        Handles formats like: energy,fluo_element_0,...,Ipreslit,IpreKB,It,Iref,fluo_DT_corrected
        """
        if len(lines) < 2:
            return None, None, {}
        
        first = lines[0].strip()
        if not first or first.startswith('#') or ',' not in first:
            return None, None, {}
        
        if 'energy' not in first.lower():
            return None, None, {}
        
        cols = [c.strip() for c in first.split(',')]
        col_roles = {}
        energy_col = -1
        
        for i, col_name in enumerate(cols):
            role = self._classify_column(col_name)
            if role in ('energy', 'energy_fbk', 'energy_setpoint', 'energy_setting'):
                if 'energy' not in col_roles:
                    col_roles['energy'] = i
                    energy_col = i
            elif role == 'i0' and 'i0' not in col_roles:
                col_roles['i0'] = i
            elif role == 'itrans' and 'itrans' not in col_roles:
                col_roles['itrans'] = i
            elif role == 'ifluor' and 'ifluor' not in col_roles:
                col_roles['ifluor'] = i
            elif role == 'mu' and 'mu' not in col_roles:
                col_roles['mu'] = i
            elif role == 'irefer' and 'irefer' not in col_roles:
                col_roles['irefer'] = i
        
        if energy_col == -1:
            return None, None, {}
        
        has_mu = 'mu' in col_roles
        has_detectors = 'i0' in col_roles and ('itrans' in col_roles or 'ifluor' in col_roles)
        if not has_mu and not has_detectors:
            return None, None, {}

        return self._collect_spectra(lines, 1, energy_col, col_roles, 'csv')

    # ─── Simple XY fallback ──────────────────────────────────────────

    def _parse_simple_xy(self, lines):
        """Parses simple 2-column text files (Energy, Mu)."""
        energies = []
        mus = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    energies.append(float(parts[0]))
                    mus.append(float(parts[1]))
                except (ValueError, IndexError):
                    continue
        
        if not energies:
            return None, None, {}
        return energies, mus, {}
