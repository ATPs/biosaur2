# Example input and outputs

This directory contains a small, real mzML input and retained regression
fixtures for TSV and Parquet output. Automated integration tests regenerate
outputs from the input without committing additional large files.

## Input provenance

`PXD010154_1554451_middle.mzML.gz` was extracted from
`PXD010154/1554451.mzML.gz` in the PRIDE archive. The original local source was:

```text
/data2/pub/proteome/PRIDE/mzML/2019/07/PXD010154/1554451.mzML.gz
```

The example is the zero-based spectrum range `[18900, 19261)` from the
38,159-spectrum source file. It contains 361 spectra over 0.973 minutes:

| Property | Value |
| --- | ---: |
| MS1 spectra | 117 |
| MS2 spectra | 244 |
| First spectrum ID | `scan=18901` |
| Last spectrum ID | `scan=19261` |
| Retention-time range | 52.453313-53.426176 minutes |

The spectrum `index` attributes were renumbered from 0 to 360. Original
spectrum IDs, scan numbers, binary peak arrays, and precursor references were
preserved. The range starts with an MS1 scan and ends at a complete acquisition
cycle, so every MS2 precursor reference remains inside the example.

## Generating the outputs

The examples were generated in the conda base environment with one worker for
reproducibility:

```bash
biosaur2 PXD010154_1554451_middle.mzML.gz \
  --workers 1 --write-hills --write-ms1 --format tsv

biosaur2 PXD010154_1554451_middle.mzML.gz \
  --workers 1 --write-hills --write-ms1 --format parquet
```

A compressed input produces `*.features`, `*.hills`, and `*.ms1` names without
the redundant `.mzML` component. Feature/hill scalar RT remains in minutes;
MS1 RT and hill point RT use seconds. Parquet feature output is one compact
file and does not create normalized sidecars or a manifest.

## Output files

| Files | Rows | Columns | Contents |
| --- | ---: | ---: | --- |
| `*.features.tsv`, `*.features.parquet` | data-dependent | 19 | Detected isotope features, monoisotopic hill traces, `feature_idx`, and `area_sum` |
| `*.hills.tsv`, `*.hills.parquet` | data-dependent | 16 | Chromatographic hills, point arrays including point RT, and feature assignment |
| `*.ms1.tsv`, `*.ms1.parquet` | 117 | 3 | Original MS1 `scan_id`, retention time in seconds, and total intensity |

Parquet files use Zstandard compression and biosaur2's default FLOAT32/narrow
integer representation, so small floating-point differences from TSV are
expected. In hills output, `feature_idx = -1` means that the hill was not
assigned to a detected feature. Feature and hill retention-time columns are in
minutes; the MS1 `RT` column is in seconds.

Selected columns from the first three TSV feature rows:

```text
feature_idx  massCalib   rtApex  intensityApex  charge  nIsotopes         mz
          1  1743.77962  52.95659    1736872.125       3          7  582.26715
          2  3434.66250  52.70979     704392.500       3          7 1145.89478
          3  1193.69333  52.59703    2521934.000       2          7  597.85394
```

Selected columns from the first three TSV hill rows:

```text
hill_idx  feature_idx   rtApex  intensityApex  nScans        mz  scanApex
       1           -1  52.48818     52669.78906       9 360.01639     18915
       2           -1  52.60733     63698.02734      10 360.01654     18957
       3           -1  52.46841     41312.47266       7 360.05295     18906
```

The first three MS1 rows are:

```text
scan_id         RT  total_intensity
  18901 3147.19878        276014660
  18902 3147.56616        266168240
  18906 3148.10478        322453410
```

Parquet output can be inspected with pandas:

```python
import pandas as pd

features = pd.read_parquet("PXD010154_1554451_middle.features.parquet")
hills = pd.read_parquet("PXD010154_1554451_middle.hills.parquet")
ms1 = pd.read_parquet("PXD010154_1554451_middle.ms1.parquet")

print(features.head())
```

Both hills formats can also be used as biosaur2 input. The committed regression
fixtures store hill point RT in minutes, so they require an explicit unit
override:

```bash
biosaur2 PXD010154_1554451_middle.hills.tsv --input-rt-unit minutes
biosaur2 PXD010154_1554451_middle.hills.parquet --input-rt-unit minutes
```
