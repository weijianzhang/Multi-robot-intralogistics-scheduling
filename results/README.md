# Publication Artifacts

No pre-existing numerical outputs were copied during packaging. Experiments write
to their original output-directory names, documented in the root README.

Use this directory for curated, verified publication artifacts, for example:

```text
results/
|-- original_comparison/
|-- common_sequencing_comparison/
`-- figures/
```

Include a provenance note with source experiment ID, implementation, decoder,
configuration, source checksum/commit, and aggregation rule. Preserve raw values
and actual timing alongside any averages. Keep original and HE-enhanced outputs
distinct. Packaging alone does not validate or reproduce historical tables.
