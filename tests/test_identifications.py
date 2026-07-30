import bz2
import gzip

import pytest

from biosaur2.identifications import (
    IdentificationRecord,
    map_identifications_to_ms2,
    parse_psm_identity,
    read_percolator_tsv,
)


HEADER = "PSMId\tscore\tq-value\tposterior_error_prob\tpeptide\tproteinIds\n"


def test_safe_psm_identity_parsing_preserves_underscores_in_run_id():
    assert parse_psm_identity("run_with_under_scores_42_3_2") == (
        "run_with_under_scores",
        42,
        3,
        2,
    )
    assert parse_psm_identity("not_parseable") is None


@pytest.mark.parametrize(
    ("suffix", "encode", "expected_compression"),
    [
        (".tsv", lambda value: value, "plain"),
        (".unexpected", gzip.compress, "gzip"),
        (".data", bz2.compress, "bzip2"),
    ],
)
def test_reader_detects_compression_by_magic_and_keeps_rank_above_one(
    tmp_path, suffix, encode, expected_compression
):
    content = (
        HEADER
        + "run_name_42_2_3\t2.0\t0.005\t0.01\tPEPTIDE\tP1\n"
        + "run_name_43_2_1\t1.0\t0.02\t0.03\tOTHER\tP2\n"
    ).encode()
    source = tmp_path / ("psms" + suffix)
    source.write_bytes(encode(content))

    result = read_percolator_tsv(source)

    assert result.qc.compression == expected_compression
    assert result.qc.row_count == 2
    assert result.qc.accepted_count == 1
    assert result.qc.rejected_q_value == 1
    assert result.records[0].parsed_rank == 3
    assert result.records[0].mapping_method == "psm_id_right_split"


def test_reader_detects_bom_semicolon_aliases_and_explicit_scan(tmp_path):
    source = tmp_path / "psms.txt"
    source.write_bytes(
        ("\ufeffspectrum;QValue;modified peptide;scan number;precursor charge\n"
         "opaque;0.001;PEPTIDE;55;3\n").encode("utf-8")
    )
    result = read_percolator_tsv(source)
    record = result.records[0]
    assert result.qc.encoding == "utf-8-sig"
    assert result.qc.delimiter == ";"
    assert record.mapping_method == "scan_column"
    assert record.parsed_scan == 55
    assert record.parsed_charge == 3


def test_reader_rejects_ambiguous_aliases_and_invalid_probabilities(tmp_path):
    ambiguous = tmp_path / "ambiguous.tsv"
    ambiguous.write_text(
        "PSMId\tspectrum\tq-value\tpeptide\nrun_1_2_1\tx\t0.1\tPEP\n"
    )
    with pytest.raises(ValueError, match="ambiguous columns for psm_id"):
        read_percolator_tsv(ambiguous)

    invalid = tmp_path / "invalid.tsv"
    invalid.write_text(HEADER + "run_1_2_1\t1\t2.0\t0.1\tPEP\tP1\n")
    result = read_percolator_tsv(invalid)
    assert result.qc.failed_rows == 1
    assert not result.records


def _record(psm_id="run_42_2_1", scan=42, charge=2, native_id=None):
    return IdentificationRecord(
        source_row=2,
        psm_id_raw=psm_id,
        score=1.0,
        q_value=0.001,
        pep=0.002,
        peptide_raw="PEPTIDE",
        proteins=None,
        parsed_run="run",
        parsed_scan=scan,
        parsed_charge=charge,
        parsed_rank=1,
        native_id=native_id,
        mapping_method="psm_id_right_split",
        mapping_status="parsed",
    )


def test_mapping_prefers_native_id_then_scan_and_flags_charge_disagreement():
    events = [
        {"ms2_event_id": 3, "native_id": "controller scan=42", "native_scan_number": 42, "charge": 2},
        {"ms2_event_id": 4, "native_id": "controller scan=43", "native_scan_number": 43, "charge": 3},
    ]
    result = map_identifications_to_ms2(
        [_record(native_id="controller scan=42"), _record(scan=43, charge=2)],
        events,
        run_id="run",
        max_unmapped_fraction=1.0,
    )
    assert result.rows[0].mapping_method == "native_id"
    assert result.rows[0].mapping_status == "mapped"
    assert result.rows[1].mapping_status == "charge_mismatch"
    assert result.rows[1].charge_agreement is False


def test_mapping_fails_visibly_when_accepted_psms_are_mostly_unmapped():
    with pytest.raises(ValueError, match="mapping appears broken") as error:
        map_identifications_to_ms2(
            [_record(), _record(psm_id="run_99_2_1", scan=99)],
            [{"ms2_event_id": 1, "native_scan_number": 42, "charge": 2}],
            run_id="run",
            max_unmapped_fraction=0.1,
        )
    assert error.value.mapping_result.unmapped_count == 1
