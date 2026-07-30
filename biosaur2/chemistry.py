"""Canonical peptidoforms, formulas and formula-specific isotope assays."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from importlib.resources import files
import re
from typing import Mapping, Optional, Sequence


PROTON_MASS = 1.007276466621
C13_C12_MASS_DIFF = 1.00335483507

MONOISOTOPIC_MASS = {
    "H": 1.00782503223,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "S": 31.9720711744,
    "P": 30.97376199842,
    "Se": 79.9165218,
}

# Polymer-residue formulas. H2O is added once for a neutral peptide.
RESIDUE_FORMULAS = {
    "A": {"C": 3, "H": 5, "N": 1, "O": 1},
    "R": {"C": 6, "H": 12, "N": 4, "O": 1},
    "N": {"C": 4, "H": 6, "N": 2, "O": 2},
    "D": {"C": 4, "H": 5, "N": 1, "O": 3},
    "C": {"C": 3, "H": 5, "N": 1, "O": 1, "S": 1},
    "E": {"C": 5, "H": 7, "N": 1, "O": 3},
    "Q": {"C": 5, "H": 8, "N": 2, "O": 2},
    "G": {"C": 2, "H": 3, "N": 1, "O": 1},
    "H": {"C": 6, "H": 7, "N": 3, "O": 1},
    "I": {"C": 6, "H": 11, "N": 1, "O": 1},
    "L": {"C": 6, "H": 11, "N": 1, "O": 1},
    "K": {"C": 6, "H": 12, "N": 2, "O": 1},
    "M": {"C": 5, "H": 9, "N": 1, "O": 1, "S": 1},
    "F": {"C": 9, "H": 9, "N": 1, "O": 1},
    "P": {"C": 5, "H": 7, "N": 1, "O": 1},
    "S": {"C": 3, "H": 5, "N": 1, "O": 2},
    "T": {"C": 4, "H": 7, "N": 1, "O": 2},
    # Selenocysteine free amino acid is C3H7NO2Se; peptide polymerization
    # removes H2O, giving the residue formula below.
    "U": {"C": 3, "H": 5, "N": 1, "O": 1, "Se": 1},
    "W": {"C": 11, "H": 10, "N": 2, "O": 1},
    "Y": {"C": 9, "H": 9, "N": 1, "O": 2},
    "V": {"C": 5, "H": 9, "N": 1, "O": 1},
}

# (nominal neutron shift, exact mass shift, natural abundance).
ISOTOPES = {
    "H": ((0, 0.0, 0.999885), (1, 1.00627674589, 0.000115)),
    "C": ((0, 0.0, 0.9893), (1, 1.00335483507, 0.0107)),
    "N": ((0, 0.0, 0.99636), (1, 0.99703489445, 0.00364)),
    "O": ((0, 0.0, 0.99757), (1, 1.00421713693, 0.00038), (2, 2.00424499329, 0.00205)),
    "S": ((0, 0.0, 0.9499), (1, 0.9993877354, 0.0075), (2, 1.9957958296, 0.0425), (4, 3.9950095356, 0.0001)),
    "P": ((0, 0.0, 1.0),),
    # The peptide mass convention uses abundant 80Se as the elemental base.
    # Natural lighter Se isotopes therefore occupy signed nominal bins.
    "Se": (
        (-6, 73.9224764 - 79.9165218, 0.0089),
        (-4, 75.9192136 - 79.9165218, 0.0937),
        (-3, 76.9199140 - 79.9165218, 0.0763),
        (-2, 77.9173091 - 79.9165218, 0.2377),
        (0, 0.0, 0.4961),
        (2, 81.9166994 - 79.9165218, 0.0873),
    ),
}


@dataclass(frozen=True)
class UnimodEntry:
    accession: str
    title: str
    aliases: tuple[str, ...]
    mono_mass: float
    average_mass: Optional[float]
    composition: Mapping[str, int]
    specificity: tuple[str, ...]


@dataclass(frozen=True)
class UnimodSnapshot:
    snapshot: str
    source: str
    attribution: str
    accessed: str
    sha256: str
    entries: Mapping[str, UnimodEntry]
    aliases: Mapping[str, str]


@dataclass(frozen=True)
class ModificationRecord:
    source_notation: str
    accession: Optional[str]
    name: Optional[str]
    site: Optional[str]
    position: Optional[int]
    position_type: str
    monoisotopic_mass_delta: Optional[float]
    average_mass_delta: Optional[float]
    elemental_delta: Optional[Mapping[str, int]]
    resolution_status: str
    fixed: bool = False


@dataclass(frozen=True)
class Peptidoform:
    original: str
    sequence: Optional[str]
    canonical: Optional[str]
    modifications: tuple[ModificationRecord, ...]
    formula: Optional[Mapping[str, int]]
    monoisotopic_mass: Optional[float]
    formula_status: str
    issues: tuple[str, ...]


@dataclass(frozen=True)
class IsotopePeak:
    isotope_index: int
    probability: float
    relative_abundance: float
    neutral_mass_shift: float
    centroid_mass_shift: float
    mz: float


@dataclass(frozen=True)
class FixedModification:
    target: str
    accession: str


def load_unimod_snapshot() -> UnimodSnapshot:
    resource = files("biosaur2").joinpath("data/unimod_subset_2026-07-29.json")
    raw = resource.read_bytes()
    payload = json.loads(raw)
    entries = {}
    aliases = {}
    for source in payload["entries"]:
        entry = UnimodEntry(
            accession=source["accession"].upper(),
            title=source["title"],
            aliases=tuple(source.get("aliases", ())),
            mono_mass=float(source["mono_mass"]),
            average_mass=None if source.get("average_mass") is None else float(source["average_mass"]),
            composition={key: int(value) for key, value in source["composition"].items()},
            specificity=tuple(source.get("specificity", ())),
        )
        entries[entry.accession] = entry
        for value in (entry.accession, entry.title, *entry.aliases):
            aliases[value.casefold()] = entry.accession
    return UnimodSnapshot(
        snapshot=payload["snapshot"],
        source=payload["source"],
        attribution=payload["attribution"],
        accessed=payload["accessed"],
        sha256=hashlib.sha256(raw).hexdigest(),
        entries=entries,
        aliases=aliases,
    )


UNIMOD = load_unimod_snapshot()


def parse_fixed_modification(value: str) -> FixedModification:
    try:
        target, modification = (part.strip() for part in value.split("=", 1))
    except ValueError as exc:
        raise ValueError("fixed modification must use SITE=MOD syntax") from exc
    target_aliases = {
        "n_term": "peptide_n_term",
        "n-term": "peptide_n_term",
        "peptide_n_term": "peptide_n_term",
        "c_term": "peptide_c_term",
        "c-term": "peptide_c_term",
        "peptide_c_term": "peptide_c_term",
        "protein_n_term": "protein_n_term",
        "protein-n-term": "protein_n_term",
        "protein_c_term": "protein_c_term",
        "protein-c-term": "protein_c_term",
    }
    normalized_target = target_aliases.get(target.casefold(), target.upper())
    if normalized_target not in {
        *RESIDUE_FORMULAS,
        "peptide_n_term",
        "peptide_c_term",
        "protein_n_term",
        "protein_c_term",
    }:
        raise ValueError("unsupported fixed-modification target: %s" % target)
    accession = _resolve_accession(modification)
    if accession is None or accession not in UNIMOD.entries:
        raise ValueError("fixed modification is absent from the pinned Unimod snapshot: %s" % modification)
    if normalized_target.startswith("protein_"):
        raise ValueError(
            "protein-terminal fixed modifications require explicit protein-terminal context"
        )
    return FixedModification(normalized_target, accession)


def _resolve_accession(value: str) -> Optional[str]:
    match = re.fullmatch(r"(?i)(?:U|UNIMOD):(\d+)", value.strip())
    if match:
        return "UNIMOD:%d" % int(match.group(1))
    return UNIMOD.aliases.get(value.strip().casefold())


def _modification(
    notation: str,
    site: Optional[str],
    position: Optional[int],
    position_type: str,
    *,
    fixed: bool = False,
) -> ModificationRecord:
    accession = _resolve_accession(notation)
    if accession is not None and accession in UNIMOD.entries:
        entry = UNIMOD.entries[accession]
        return ModificationRecord(
            notation,
            accession,
            entry.title,
            site,
            position,
            position_type,
            entry.mono_mass,
            entry.average_mass,
            entry.composition,
            "exact",
            fixed,
        )
    mass_match = re.fullmatch(r"[+-](?:\d+(?:\.\d*)?|\.\d+)", notation.strip())
    if mass_match:
        return ModificationRecord(
            notation,
            None,
            None,
            site,
            position,
            position_type,
            float(notation),
            None,
            None,
            "mass_only",
            fixed,
        )
    return ModificationRecord(
        notation,
        accession,
        None,
        site,
        position,
        position_type,
        None,
        None,
        None,
        "unavailable",
        fixed,
    )


def _consume_modification(text: str, start: int):
    opener = text[start]
    closer = "]" if opener == "[" else ")"
    end = text.find(closer, start + 1)
    if end < 0:
        raise ValueError("unclosed modification annotation")
    notation = text[start + 1 : end].strip()
    if not notation:
        raise ValueError("empty modification annotation")
    if opener == "(" and _resolve_accession(notation) is None:
        raise ValueError("parenthesized modifications must identify a pinned Unimod entry")
    return notation, end + 1


def _add_formula(target, source):
    for element, count in source.items():
        target[element] = target.get(element, 0) + int(count)


def formula_mass(formula: Mapping[str, int]) -> float:
    try:
        return float(sum(MONOISOTOPIC_MASS[element] * count for element, count in formula.items()))
    except KeyError as exc:
        raise ValueError("unsupported formula element: %s" % exc.args[0]) from exc


def parse_peptidoform(
    value: str,
    *,
    fixed_modifications: Sequence[FixedModification | str] = (),
) -> Peptidoform:
    original = value
    text = value.strip()
    flank = re.fullmatch(r"([A-Z])\.(.+)\.([A-Z])", text)
    if flank:
        text = flank.group(2)
    issues = []
    modifications = []

    nterm_notation = None
    if text.startswith("["):
        try:
            nterm_notation, end = _consume_modification(text, 0)
        except ValueError as exc:
            return Peptidoform(original, None, None, (), None, None, "unavailable", (str(exc),))
        if end >= len(text) or text[end] != "-":
            return Peptidoform(original, None, None, (), None, None, "unavailable", ("leading modification lacks peptide-terminal '-'",))
        text = text[end + 1 :]

    cterm_notation = None
    match = re.search(r"-\[([^][]+)\]$", text)
    if match:
        cterm_notation = match.group(1).strip()
        text = text[: match.start()]

    sequence = []
    index = 0
    try:
        while index < len(text):
            residue = text[index]
            if residue not in RESIDUE_FORMULAS:
                raise ValueError("unsupported residue or notation at character %d" % (index + 1))
            position = len(sequence)
            sequence.append(residue)
            index += 1
            while index < len(text) and text[index] in "[(":
                notation, index = _consume_modification(text, index)
                modifications.append(
                    _modification(notation, residue, position, "residue")
                )
    except ValueError as exc:
        return Peptidoform(original, None, None, tuple(modifications), None, None, "unavailable", (str(exc),))
    if not sequence:
        return Peptidoform(original, None, None, (), None, None, "unavailable", ("empty peptide sequence",))

    sequence_text = "".join(sequence)
    if nterm_notation is not None:
        modifications.append(
            _modification(nterm_notation, None, None, "peptide_n_term")
        )
    if cterm_notation is not None:
        modifications.append(
            _modification(cterm_notation, None, None, "peptide_c_term")
        )

    fixed = [
        item if isinstance(item, FixedModification) else parse_fixed_modification(item)
        for item in fixed_modifications
    ]
    for rule in fixed:
        positions = []
        if rule.target in RESIDUE_FORMULAS:
            positions = [
                (position, residue, "residue")
                for position, residue in enumerate(sequence_text)
                if residue == rule.target
            ]
        elif rule.target in {"peptide_n_term", "peptide_c_term"}:
            positions = [(None, None, rule.target)]
        for position, site, position_type in positions:
            if any(
                mod.accession == rule.accession
                and mod.position == position
                and mod.position_type == position_type
                for mod in modifications
            ):
                continue
            modifications.append(
                _modification(rule.accession, site, position, position_type, fixed=True)
            )

    status = "exact"
    if any(mod.resolution_status == "unavailable" for mod in modifications):
        status = "unavailable"
    elif any(mod.resolution_status == "mass_only" for mod in modifications):
        status = "mass_only"

    formula = {"H": 2, "O": 1}
    for residue in sequence_text:
        _add_formula(formula, RESIDUE_FORMULAS[residue])
    if status == "exact":
        for mod in modifications:
            _add_formula(formula, mod.elemental_delta or {})
        if any(count < 0 for count in formula.values()):
            issues.append("modifications produce a negative elemental count")
            status = "unavailable"
    exact_formula = formula if status == "exact" else None
    if status == "unavailable":
        mass = None
    elif status == "exact":
        mass = formula_mass(formula)
    else:
        base_formula = {"H": 2, "O": 1}
        for residue in sequence_text:
            _add_formula(base_formula, RESIDUE_FORMULAS[residue])
        mass = formula_mass(base_formula) + sum(
            mod.monoisotopic_mass_delta or 0.0 for mod in modifications
        )

    by_position = {}
    nterm = []
    cterm = []
    for mod in modifications:
        token = "[%s]" % (mod.accession or ("%+.6f" % mod.monoisotopic_mass_delta if mod.monoisotopic_mass_delta is not None else mod.source_notation))
        if mod.position_type == "peptide_n_term":
            nterm.append(token)
        elif mod.position_type == "peptide_c_term":
            cterm.append(token)
        else:
            by_position.setdefault(mod.position, []).append(token)
    canonical = "".join(sorted(nterm))
    if nterm:
        canonical += "-"
    canonical += "".join(
        residue + "".join(sorted(by_position.get(position, ())))
        for position, residue in enumerate(sequence_text)
    )
    if cterm:
        canonical += "-" + "".join(sorted(cterm))
    return Peptidoform(
        original,
        sequence_text,
        canonical,
        tuple(modifications),
        exact_formula,
        mass,
        status,
        tuple(issues),
    )


def precursor_mz(neutral_mass: float, charge: int) -> float:
    if charge <= 0:
        raise ValueError("charge must be positive")
    return (float(neutral_mass) + charge * PROTON_MASS) / charge


def isotope_library(
    formula: Mapping[str, int],
    charge: int,
    *,
    max_isotopes: int = 8,
) -> tuple[IsotopePeak, ...]:
    """Return a truncated natural-abundance nominal isotope distribution.

    Exact isotope mass defects are probability-weighted within each nominal
    neutron bin; elemental counts and abundances remain formula-specific.
    """

    if max_isotopes < 1:
        raise ValueError("max_isotopes must be positive")
    if charge <= 0:
        raise ValueError("charge must be positive")
    probabilities = {0: 1.0}
    mass_sums = {0: 0.0}
    for element in sorted(formula):
        count = int(formula[element])
        if count < 0:
            raise ValueError("formula contains a negative elemental count")
        isotope_states = ISOTOPES.get(element)
        if isotope_states is None:
            raise ValueError("no isotope model for element %s" % element)
        for _ in range(count):
            next_probability = {}
            next_mass_sum = {}
            for current in sorted(probabilities):
                current_probability = probabilities[current]
                for nominal_shift, exact_shift, abundance in isotope_states:
                    destination = current + nominal_shift
                    if destination >= max_isotopes:
                        continue
                    next_probability[destination] = next_probability.get(
                        destination, 0.0
                    ) + current_probability * abundance
                    next_mass_sum[destination] = next_mass_sum.get(
                        destination, 0.0
                    ) + (
                        mass_sums[current] * abundance
                        + current_probability * abundance * exact_shift
                    )
            probabilities, mass_sums = next_probability, next_mass_sum
    base_mass = formula_mass(formula)
    maximum = max(probabilities.values())
    peaks = []
    indices = sorted(probabilities, key=lambda index: (index != 0, index))
    for index in indices:
        probability = probabilities[index]
        centroid_shift = 0.0 if probability == 0 else mass_sums[index] / probability
        # The assay grid follows the conventional monoisotopic 13C spacing used
        # by precursor selection/search engines. The formula-specific natural
        # abundance centroid is retained separately instead of silently moving
        # the nominal M+n assay target away from that interoperable grid.
        shift = (
            centroid_shift
            if formula.get("Se", 0)
            else index * C13_C12_MASS_DIFF
        )
        peaks.append(
            IsotopePeak(
                isotope_index=index,
                probability=probability,
                relative_abundance=0.0 if maximum == 0 else probability / maximum,
                neutral_mass_shift=shift,
                centroid_mass_shift=centroid_shift,
                mz=precursor_mz(base_mass + shift, charge),
            )
        )
    return tuple(peaks)
