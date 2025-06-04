# Unit tests for core functions in starter_geneturing
from starter_geneturing import (
    disease_gene_location,
    exact_match,
    gene_disease_association,
    get_answer,
    human_genome_dna_alignment,
)


def test_exact_match() -> None:
    """Test exact string matching."""
    assert exact_match("chr1", "chr1") == 1.0
    assert exact_match("chr1", "chr2") == 0.0


def test_gene_disease_association() -> None:
    """Test gene-disease association matching with set overlap."""
    pred = ["BRCA1", "TP53"]
    true = ["TP53"]
    assert gene_disease_association(pred, true) == 1.0


def test_disease_gene_location() -> None:
    """Test disease-gene location comparison using set overlap."""
    pred = ["chr1:100-200", "chr2:300-400"]
    true = ["chr1:100-200"]
    assert disease_gene_location(pred, true) == 1.0


def test_human_genome_dna_alignment() -> None:
    """Test alignment score between genome region predictions."""
    assert human_genome_dna_alignment("chr1:100-200", "chr1:100-200") == 1.0
    assert human_genome_dna_alignment("chr1:123-456", "chr1:789-999") == 0.5
    assert human_genome_dna_alignment("chr2:123-456", "chr1:789-999") == 0.0


def test_get_answer_snplocation() -> None:
    """Test answer normalization for SNP location task."""
    ans = "SNP rs123 is located on 1"
    assert get_answer(ans, "SNP location") == "chr1"


def test_get_answer_protein_coding() -> None:
    """Test binary answer conversion for protein-coding gene task."""
    assert get_answer("Answer: Yes", "Protein-coding genes") == "TRUE"
    assert get_answer("Answer: No", "Protein-coding genes") == "NA"
