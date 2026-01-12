import numpy as np
import pandas as pd
import re
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

from rdkit import Chem

pt = Chem.GetPeriodicTable()


E_h = 4.35974434e-18

Na = 6.02214179e23

E_h_to_kJ_mol = E_h * Na / 1000  # Hartree to kJ/mol conversion factor


def _to_float(token: str) -> Optional[float]:
    try:
        return float(token.replace('D', 'E'))
    except Exception:
        return None


def _last_float_in_line(line: str) -> Optional[float]:
    parts = line.split()
    for tok in reversed(parts):
        val = _to_float(tok)
        if val is not None:
            return val
    return None


def _float_at_index(line: str, idx: int) -> Optional[float]:
    parts = line.split()
    if len(parts) > idx:
        return _to_float(parts[idx])
    return None


def _extract_scf_done(line: str) -> Optional[float]:
    # Example: "SCF Done:  E(RB3LYP) =  -232.123456789     A.U. after ..."
    m = re.search(r'E\([^)]+\)\s*=\s*([-+]?\d+\.\d+)', line)
    return _to_float(m.group(1)) if m else None


def _extract_archive_value(lines: list[str], i: int, key: str) -> Optional[float]:
    """
    Extract a value from Gaussian archive-style chunks, which may wrap.
    """
    cur = lines[i].strip()
    nxt = lines[i + 1].strip() if i + 1 < len(lines) else ''
    joined = cur + nxt
    needle = f'\\{key}='
    start = joined.find(needle)
    if start == -1:
        return None
    start += len(needle)
    end = joined.find('\\', start)
    if end == -1:
        return None
    return _to_float(joined[start:end])


def _parse_gaussian_cpu_time(line: str) -> Optional[float]:
    """
    Parse:
    "Job cpu time:       2 days  0 hours 46 minutes 40.0 seconds."
    Return seconds (float).
    """
    m = re.search(
        r"Job cpu time:\s*"
        r"(?:(\d+)\s+days?\s+)?"
        r"(?:(\d+)\s+hours?\s+)?"
        r"(?:(\d+)\s+minutes?\s+)?"
        r"(?:(\d+(?:\.\d+)?)\s+seconds?)",
        line
    )
    if not m:
        return None

    days = int(m.group(1)) if m.group(1) else 0
    hours = int(m.group(2)) if m.group(2) else 0
    minutes = int(m.group(3)) if m.group(3) else 0
    seconds = float(m.group(4)) if m.group(4) else 0.0

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _parse_gaussian_termination_datetime(line: str) -> Optional[datetime]:
    """
    Parse:
    "Normal termination of Gaussian 09 at Mon Jan 12 05:43:39 2026."
    """
    m = re.search(r"Normal termination of Gaussian\s+\d+\s+at\s+(.+)\.\s*$", line)
    if not m:
        return None
    s = m.group(1).strip()
    # Example format: "Mon Jan 12 05:43:39 2026"
    return datetime.strptime(s, "%a %b %d %H:%M:%S %Y")


class GaussianParser:

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.lines = self._read_log_file(log_file)

        self._resources = None
        self._route_section = None

        self._geometry = None
        self._frequencies = None
        self._e_elect = None
        self._zpe_correction = None

        self._timing = None
        self._cpu_time_seconds = None
        self._real_time_seconds = None

    def _read_log_file(self, log_file: str) -> List[str]:
        with open(log_file, 'r') as f:
            return f.readlines()


    @property
    def real_time_seconds(self) -> Optional[float]:
        """
        Estimate wall time as cpu_time / nprocshared.
        Returns None if either piece is missing.
        """
        if self._real_time_seconds is not None:
            return self._real_time_seconds

        cpu = self.timing.get("cpu_time_seconds") if self.timing else None
        nproc = self.resources.get("nprocshared") if self.resources else None

        if cpu is None or not nproc:
            self._real_time_seconds = None
        else:
            self._real_time_seconds = cpu / float(nproc)

        return self._real_time_seconds

    @property
    def real_time_hms(self) -> Optional[str]:
        s = self.real_time_seconds
        if s is None:
            return None
        s_int = int(round(s))
        h = s_int // 3600
        m = (s_int % 3600) // 60
        sec = s_int % 60
        return f"{h}h {m}m {sec}s"


    @property
    def timing(self) -> Dict[str, Any]:
        if self._timing is None:
            self._timing = self._parse_timing()
        return self._timing

    @property
    def cpu_time_seconds(self) -> Optional[float]:
        if self._cpu_time_seconds is None:
            t = self.timing.get("cpu_time_seconds")
            self._cpu_time_seconds = t
        return self._cpu_time_seconds

    @property
    def resources(self) -> Dict[str, Any]:
        if self._resources is None:
            self._resources = self._parse_resources()
        return self._resources

    @property
    def route_section(self) -> Optional[str]:
        if self._route_section is None:
            self._route_section = self._parse_route_section()
        return self._route_section

    @property
    def geometry(self) -> np.ndarray:
        if self._geometry is None:
            self._geometry = self._parse_geometry()
        return self._geometry

    @property
    def frequencies(self) -> List[float]:
        if self._frequencies is None:
            self._frequencies = self._parse_frequencies()
        return self._frequencies

    @property
    def e_elect(self) -> float:
        if self._e_elect is None:
            self._e_elect = self._parse_e_elect()
        return self._e_elect

    @property
    def zpe_correction(self) -> float:
        if self._zpe_correction is None:
            self._zpe_correction = self._parse_zpe_correction()
        return self._zpe_correction

    @property
    def opt_cycles(self) -> Optional[int]:
        if getattr(self, "_opt_cycles", None) is None:
            self._opt_cycles = self._parse_opt_cycles()
        return self._opt_cycles

    def _parse_opt_cycles(self) -> Optional[int]:
        # --- Primary: read "Step number" lines and take the maximum ---
        step_nums = []
        step_re = re.compile(r"\bStep number\s+(\d+)\b", re.IGNORECASE)

        for line in self.lines:
            m = step_re.search(line)
            if m:
                step_nums.append(int(m.group(1)))

        if step_nums:
            return max(step_nums)

        # --- Fallback: count convergence-table occurrences ---
        # Typically each opt cycle prints a block containing these lines.
        # Using "Maximum Force" is a good anchor.
        n_blocks = 0
        for line in self.lines:
            if line.strip().startswith("Maximum Force"):
                n_blocks += 1

        if n_blocks:
            return n_blocks

        # If the job isn't an optimization (or log is truncated), return None
        return None

    def _parse_resources(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "nprocshared": None,   # int
            "mem_mb": None,        # int
            "chk": None,           # str
        }

        # scan early part of file; directives are near the top
        max_scan = min(len(self.lines), 500)
        for line in self.lines[:max_scan]:
            s = line.strip()

            # %nprocshared=8
            m = re.match(r"%\s*nprocshared\s*=\s*(\d+)", s, flags=re.IGNORECASE)
            if m:
                out["nprocshared"] = int(m.group(1))
                continue

            # %mem=32768MB   or %mem=32GB
            m = re.match(r"%\s*mem\s*=\s*([0-9]+)\s*([A-Za-z]+)?", s, flags=re.IGNORECASE)
            if m:
                val = int(m.group(1))
                unit = (m.group(2) or "").upper()

                if unit in ("", "MB"):
                    out["mem_mb"] = val
                elif unit == "GB":
                    out["mem_mb"] = val * 1024
                elif unit == "KB":
                    out["mem_mb"] = val // 1024
                else:
                    # unknown unit; keep raw MB guess
                    out["mem_mb"] = val
                continue

            # %chk=check.chk
            m = re.match(r"%\s*chk\s*=\s*(\S+)", s, flags=re.IGNORECASE)
            if m:
                out["chk"] = m.group(1)
                continue

        return out

    def _parse_route_section(self) -> Optional[str]:
        max_scan = min(len(self.lines), 800)

        dash_idx = []
        for i, line in enumerate(self.lines[:max_scan]):
            if line.strip().startswith('-') and len(line.strip()) >= 10:
                dash_idx.append(i)

        # Need two dashed lines with content between them
        for a, b in zip(dash_idx, dash_idx[1:]):
            between = self.lines[a+1:b]
            # Look for a route line starting with #
            if any(l.lstrip().startswith('#') for l in between):
                # Keep only non-empty lines, preserve spacing a bit
                route_lines = [l.strip() for l in between if l.strip()]
                # Join with spaces to undo wrapping
                route = " ".join(route_lines)
                route = re.sub(r"\s+", " ", route).strip()
                # Join things like "wb97xd/def2 tzvp" → "wb97xd/def2tzvp"
                route = re.sub(
                    r"([A-Za-z0-9\-\+]+)/\s*([A-Za-z0-9\-\+]+)\s+([A-Za-z0-9\-\+]+)",
                    r"\1/\2\3",
                    route
                )

                return route

        return None

    def parse_method_basis(route: str) -> tuple[Optional[str], Optional[str]]:
        # find token like "wb97xd/def2tzvp"
        m = re.search(r"\b([A-Za-z0-9\-\+\(\)]+)\s*/\s*([A-Za-z0-9\-\+]+)\b", route)
        if not m:
            return None, None
        return m.group(1), m.group(2)

    def _parse_timing(self) -> Dict[str, Any]:
        cpu_seconds = None
        termination_dt = None
        terminated_normally = False

        # Search from the bottom since these lines are at the end
        for line in reversed(self.lines[-4000:]):  # last chunk is enough
            if cpu_seconds is None and "Job cpu time:" in line:
                cpu_seconds = _parse_gaussian_cpu_time(line)
            if termination_dt is None and "Normal termination of Gaussian" in line:
                termination_dt = _parse_gaussian_termination_datetime(line)
                if termination_dt is not None:
                    terminated_normally = True

            if cpu_seconds is not None and termination_dt is not None:
                break

        return {
            "terminated_normally": terminated_normally,
            "termination_datetime": termination_dt,      # datetime or None
            "cpu_time_seconds": cpu_seconds,             # float or None
            "cpu_time_days": (cpu_seconds / 86400) if cpu_seconds is not None else None,
        }

    def _parse_gaussian_cpu_time(line: str) -> Optional[float]:
        """
        Parse:
        "Job cpu time:       2 days  0 hours 46 minutes 40.0 seconds."
        Return seconds (float).
        """
        m = re.search(
            r"Job cpu time:\s*"
            r"(?:(\d+)\s+days?\s+)?"
            r"(?:(\d+)\s+hours?\s+)?"
            r"(?:(\d+)\s+minutes?\s+)?"
            r"(?:(\d+(?:\.\d+)?)\s+seconds?)",
            line
        )
        if not m:
            return None

        days = int(m.group(1)) if m.group(1) else 0
        hours = int(m.group(2)) if m.group(2) else 0
        minutes = int(m.group(3)) if m.group(3) else 0
        seconds = float(m.group(4)) if m.group(4) else 0.0

        return days * 86400 + hours * 3600 + minutes * 60 + seconds

    def _parse_geometry(self) -> np.ndarray:
        # -------- 1) Try last "Standard orientation:" block --------
        for i in range(len(self.lines) - 1, -1, -1):
            if 'Standard orientation:' in self.lines[i]:
                j = i + 5  # table starts ~5 lines later
                coords = []

                while j < len(self.lines):
                    line = self.lines[j]
                    s = line.strip()

                    # stop conditions (mirrors your old version)
                    if not s:
                        break
                    if line.lstrip().startswith('---') or '-----' in line:
                        break

                    parts = line.split()
                    # only accept real atom rows
                    if parts and parts[0].isdigit() and len(parts) >= 6:
                        atomic_number = int(parts[1])
                        symbol = pt.GetElementSymbol(atomic_number)
                        x, y, z = map(float, parts[3:6])
                        coords.append((symbol, x, y, z))

                    j += 1

                if coords:
                    return np.array(coords, dtype=object)
                break  # found a Standard orientation header but couldn't parse -> fall back

        # -------- 2) Fall back to last "Input orientation:" block --------
        last_block = None
        i = 0
        while i < len(self.lines):
            if 'Input orientation:' in self.lines[i]:
                j = i + 5  # skip header
                coords = []

                while j < len(self.lines):
                    line = self.lines[j]
                    if '-----' in line:  # end of table in this format
                        break

                    parts = line.split()
                    if parts and parts[0].isdigit() and len(parts) >= 6:
                        atomic_number = int(parts[1])
                        symbol = pt.GetElementSymbol(atomic_number)
                        x, y, z = map(float, parts[3:6])
                        coords.append((symbol, x, y, z))

                    j += 1

                if coords:
                    last_block = coords  # keep overwriting to end up with the last one
                i = j
            i += 1

        if last_block:
            return np.array(last_block, dtype=object)

        raise ValueError("No parsable 'Standard orientation' or 'Input orientation' geometry blocks found.")


    def _parse_frequencies(self) -> List[float]:
        freqs_all = []
        for line in self.lines:
            if 'Frequencies --' in line:
                parts = line.split('--', 1)[1].split()
                freqs_all.extend(float(x) for x in parts)
        return freqs_all

    def _parse_e_elect(self) -> float:
        e_elect = None          # best electronic energy (Hartree)
        e0_composite = None     # composite 0 K energy (Hartree)
        scaled_zpe = None       # scaled ZPE (Hartree)

        for i, line in enumerate(self.lines):
            if 'SCF Done:' in line:
                val = _extract_scf_done(line)
                if val is not None:
                    e_elect = val

            elif ' E2(' in line and ' E(' in line:
                # e.g., some correlated energy printouts
                val = _last_float_in_line(line)
                if val is not None:
                    e_elect = val

            elif 'MP2 =' in line:
                val = _last_float_in_line(line)
                if val is not None:
                    e_elect = val

            elif 'E(CORR)=' in line:
                val = _float_at_index(line, 3)
                if val is not None:
                    e_elect = val

            elif 'CCSD(T)=' in line:
                val = _float_at_index(line, 1)
                if val is not None:
                    e_elect = val

            elif 'CBS-QB3 (0 K)' in line:
                val = _float_at_index(line, 3)
                if val is not None:
                    e0_composite = val

            elif 'E(CBS-QB3)=' in line:
                val = _float_at_index(line, 1)
                if val is not None:
                    e_elect = val

            elif 'CBS-4 (0 K)=' in line:
                val = _float_at_index(line, 3)
                if val is not None:
                    e0_composite = val

            elif 'G3(0 K)' in line:
                val = _float_at_index(line, 2)
                if val is not None:
                    e0_composite = val

            elif 'G3 Energy=' in line:
                val = _float_at_index(line, 2)
                if val is not None:
                    e_elect = val

            elif 'G4(0 K)' in line:
                val = _float_at_index(line, 2)
                if val is not None:
                    e0_composite = val

            elif 'G4 Energy=' in line:
                val = _float_at_index(line, 2)
                if val is not None:
                    e_elect = val

            elif 'G4MP2(0 K)' in line:
                val = _float_at_index(line, 2)
                if val is not None:
                    e0_composite = val

            elif 'G4MP2 Energy=' in line:
                val = _float_at_index(line, 2)
                if val is not None:
                    e_elect = val

            elif 'E(ZPE)' in line:
                val = _float_at_index(line, 1)
                if val is not None:
                    scaled_zpe = val

            elif '\\ZeroPoint=' in line:
                val = _extract_archive_value(self.lines, i, 'ZeroPoint')
                if val is not None:
                    scaled_zpe = val

            elif 'HF=' in line and e_elect is None:
                val = _extract_archive_value(self.lines, i, 'HF')
                if val is not None:
                    e_elect = val

        # Match your old return logic:
        if e0_composite is not None and scaled_zpe is not None:
            return (e0_composite - scaled_zpe) * E_h_to_kJ_mol
        if e0_composite is not None:
            return e0_composite * E_h_to_kJ_mol
        if e_elect is not None:
            return e_elect * E_h_to_kJ_mol

        raise ValueError("No electronic energy found in log file.")

    def _parse_zpe_correction(self) -> float:
        # Prefer explicit "Zero-point correction=" line
        zpe_h = None

        for i, line in enumerate(self.lines):
            if 'Zero-point correction=' in line:
                # Gaussian often: "Zero-point correction= 0.123456 (Hartree/Particle)"
                # safer: take the first float-like token after '='
                rhs = line.split('=', 1)[1]
                val = _last_float_in_line(rhs)
                if val is not None:
                    zpe_h = val

            elif '\\ZeroPoint=' in line:
                val = _extract_archive_value(self.lines, i, 'ZeroPoint')
                if val is not None:
                    zpe_h = val

        if zpe_h is None:
            raise ValueError("ZPE correction not found in log file.")

        return zpe_h * E_h_to_kJ_mol

    def print_geometry(self):
        if self.geometry is None:
            self._geometry = self._parse_geometry()
        for atom in self.geometry:
            print(f"{atom[0]:<2} {atom[1]:>10.6f} {atom[2]:>10.6f} {atom[3]:>10.6f}")
