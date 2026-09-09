AA_UNK_IDX = 21
AA_EOS_IDX = 23
CDS_SOS_IDX = 65
PAD_IDX = 0


def build_vocabularies() -> tuple[dict[str, int], dict[int, str], dict[int, list[int]]]:
    aa_vocab = {
        "A": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "Q": 14,
        "R": 15,
        "S": 16,
        "T": 17,
        "V": 18,
        "W": 19,
        "Y": 20,
        "X": AA_UNK_IDX,
        "Z": AA_UNK_IDX,
        "B": AA_UNK_IDX,
        "U": AA_UNK_IDX,
        "O": AA_UNK_IDX,
        "*": 22,
    }
    aa_to_codons = {
        "A": ["GCT", "GCC", "GCA", "GCG"],
        "C": ["TGT", "TGC"],
        "D": ["GAT", "GAC"],
        "E": ["GAA", "GAG"],
        "F": ["TTT", "TTC"],
        "G": ["GGT", "GGA", "GGC", "GGG"],
        "H": ["CAT", "CAC"],
        "I": ["ATT", "ATC", "ATA"],
        "K": ["AAA", "AAG"],
        "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
        "M": ["ATG"],
        "N": ["AAT", "AAC"],
        "P": ["CCT", "CCC", "CCA", "CCG"],
        "Q": ["CAA", "CAG"],
        "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
        "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
        "T": ["ACT", "ACC", "ACA", "ACG"],
        "V": ["GTT", "GTC", "GTA", "GTG"],
        "W": ["TGG"],
        "Y": ["TAT", "TAC"],
        "*": ["TAA", "TAG", "TGA"],
    }

    codon_vocab: dict[str, int] = {}
    next_id = 1
    for codons in aa_to_codons.values():
        for codon in codons:
            codon_vocab[codon] = next_id
            next_id += 1

    idx_to_codon = {value: key for key, value in codon_vocab.items()}
    aa_id_to_codon_ids = {
        aa_vocab[aa]: [codon_vocab[codon] for codon in codons]
        for aa, codons in aa_to_codons.items()
    }
    return aa_vocab, idx_to_codon, aa_id_to_codon_ids


def normalize_amino_acids(amino_acids: str) -> str:
    normalized = "".join(amino_acids.split()).upper()
    if not normalized:
        raise ValueError("Amino acid sequence must not be empty")
    return normalized
