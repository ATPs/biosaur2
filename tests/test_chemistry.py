import math

import pytest
from pyteomics import mass

from biosaur2.chemistry import (
    UNIMOD,
    isotope_library,
    parse_fixed_modification,
    parse_peptidoform,
    precursor_mz,
)


@pytest.mark.parametrize(
    ("source", "canonical"),
    [
        ("EEIVENPSSSASESNTSTSIVN[U:7]R", "EEIVENPSSSASESNTSTSIVN[UNIMOD:7]R"),
        ("N[UNIMOD:7]R", "N[UNIMOD:7]R"),
        ("M[U:35]", "M[UNIMOD:35]"),
        ("Q(UniMod:28)", "Q[UNIMOD:28]"),
        ("[U:1]-PEPTIDE", "[UNIMOD:1]-PEPTIDE"),
        ("[UNIMOD:1]-PEPTIDE", "[UNIMOD:1]-PEPTIDE"),
        ("K.PEPTIDE.R", "PEPTIDE"),
    ],
)
def test_required_exact_peptidoform_notations(source, canonical):
    result = parse_peptidoform(source)
    assert result.formula_status == "exact"
    assert result.canonical == canonical
    assert result.formula
    assert result.monoisotopic_mass > 0


def test_mass_only_and_unavailable_modifications_are_distinct():
    mass_only = parse_peptidoform("M[+15.994915]")
    unavailable = parse_peptidoform("M[UNIMOD:999999]")
    assert mass_only.formula_status == "mass_only"
    assert mass_only.formula is None
    assert mass_only.monoisotopic_mass is not None
    assert unavailable.formula_status == "unavailable"
    assert unavailable.monoisotopic_mass is None


def test_fixed_carbamidomethyl_is_explicit_and_not_double_applied():
    unmodified = parse_peptidoform("ACDC")
    fixed = parse_peptidoform("ACDC", fixed_modifications=["C=UNIMOD:4"])
    explicit = parse_peptidoform(
        "AC[UNIMOD:4]DC[UNIMOD:4]", fixed_modifications=["C=UNIMOD:4"]
    )
    assert fixed.monoisotopic_mass - unmodified.monoisotopic_mass == pytest.approx(
        2 * 57.021464, abs=2e-6
    )
    assert explicit.monoisotopic_mass == pytest.approx(fixed.monoisotopic_mass)
    assert sum(mod.accession == "UNIMOD:4" for mod in explicit.modifications) == 2


def test_protein_terminal_fixed_mod_requires_context():
    with pytest.raises(ValueError, match="protein-terminal"):
        parse_fixed_modification("Protein-N-term=UNIMOD:1")


def test_unmodified_formula_mass_matches_pyteomics():
    peptide = parse_peptidoform("PEPTIDE")
    assert peptide.monoisotopic_mass == pytest.approx(
        mass.calculate_mass(sequence="PEPTIDE"), abs=2e-8
    )
    assert precursor_mz(peptide.monoisotopic_mass, 2) == pytest.approx(
        mass.calculate_mass(sequence="PEPTIDE", charge=2), abs=2e-8
    )


def test_formula_specific_isotope_library_is_normalized_and_mass_ordered():
    light = parse_peptidoform("PEPTIDE")
    sulfur = parse_peptidoform("MCMCMCMCMCMC")
    light_peaks = isotope_library(light.formula, 2, max_isotopes=6)
    sulfur_peaks = isotope_library(sulfur.formula, 2, max_isotopes=6)
    assert [peak.isotope_index for peak in light_peaks] == list(range(6))
    assert all(
        right.mz > left.mz for left, right in zip(light_peaks, light_peaks[1:])
    )
    assert max(peak.relative_abundance for peak in light_peaks) == pytest.approx(1.0)
    assert light_peaks[1].neutral_mass_shift == pytest.approx(1.00335483507)
    assert math.isfinite(light_peaks[1].centroid_mass_shift)
    assert sulfur_peaks[2].relative_abundance > light_peaks[2].relative_abundance


def test_selenocysteine_has_exact_se_formula_and_bounded_isotope_library():
    peptide = parse_peptidoform("GUGC", fixed_modifications=["C=UNIMOD:4"])
    assert peptide.formula_status == "exact"
    assert peptide.formula["Se"] == 1
    assert peptide.canonical == "GUGC[UNIMOD:4]"
    assert peptide.monoisotopic_mass == pytest.approx(
        mass.calculate_mass(sequence="GUGC") + 57.021464, abs=2e-6
    )
    peaks = isotope_library(peptide.formula, 2, max_isotopes=6)
    assert peaks[0].isotope_index == 0
    assert {-6, -4, -3, -2, 0, 2} <= {
        peak.isotope_index for peak in peaks
    }
    by_index = {peak.isotope_index: peak for peak in peaks}
    assert by_index[-2].mz < by_index[0].mz < by_index[2].mz
    assert all(math.isfinite(peak.mz) for peak in peaks)
    assert all(peak.probability >= 0 for peak in peaks)


def test_pinned_snapshot_has_hash_and_required_pxd_modifications():
    assert len(UNIMOD.sha256) == 64
    assert all(character in "0123456789abcdef" for character in UNIMOD.sha256)
    assert {"UNIMOD:1", "UNIMOD:4", "UNIMOD:7", "UNIMOD:28", "UNIMOD:35"} <= set(
        UNIMOD.entries
    )
