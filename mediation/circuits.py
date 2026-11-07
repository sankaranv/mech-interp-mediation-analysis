"""The circuits the paper examines: IOI heads with their roles and Greater-Than MLPs.

IOI heads and roles are those of Wang et al. (2023) for GPT-2 small. Greater-Than MLPs are
those of Hanna et al. (2023): the early MLPs 0-3 carry the start year forward and the late
MLPs 8-11 compute the comparison.
"""

from mediation.patching import AttentionHead, MlpLayer

NAME_MOVER = "Name Mover"
BACKUP_NAME_MOVER = "Backup Name Mover"
NEGATIVE_NAME_MOVER = "Negative Name Mover"
S_INHIBITION = "S-Inhibition"
DUPLICATE_TOKEN = "Duplicate Token"
INDUCTION = "Induction"
PREVIOUS_TOKEN = "Previous Token"
OTHER = "Other"

# Role of each of the 26 circuit heads of GPT-2 small.
IOI_HEAD_ROLES: dict[AttentionHead, str] = {
    AttentionHead(9, 9): NAME_MOVER,
    AttentionHead(9, 6): NAME_MOVER,
    AttentionHead(10, 0): NAME_MOVER,
    AttentionHead(10, 7): NEGATIVE_NAME_MOVER,
    AttentionHead(11, 10): NEGATIVE_NAME_MOVER,
    AttentionHead(7, 3): S_INHIBITION,
    AttentionHead(7, 9): S_INHIBITION,
    AttentionHead(8, 6): S_INHIBITION,
    AttentionHead(8, 10): S_INHIBITION,
    AttentionHead(0, 1): DUPLICATE_TOKEN,
    AttentionHead(3, 0): DUPLICATE_TOKEN,
    AttentionHead(0, 10): DUPLICATE_TOKEN,
    AttentionHead(5, 5): INDUCTION,
    AttentionHead(6, 9): INDUCTION,
    AttentionHead(5, 8): INDUCTION,
    AttentionHead(5, 9): INDUCTION,
    AttentionHead(2, 2): PREVIOUS_TOKEN,
    AttentionHead(4, 11): PREVIOUS_TOKEN,
    AttentionHead(9, 0): BACKUP_NAME_MOVER,
    AttentionHead(9, 7): BACKUP_NAME_MOVER,
    AttentionHead(10, 1): BACKUP_NAME_MOVER,
    AttentionHead(10, 2): BACKUP_NAME_MOVER,
    AttentionHead(10, 6): BACKUP_NAME_MOVER,
    AttentionHead(10, 10): BACKUP_NAME_MOVER,
    AttentionHead(11, 2): BACKUP_NAME_MOVER,
    AttentionHead(11, 9): BACKUP_NAME_MOVER,
}

# Heads whose role Wang et al. mark as less certain.
IOI_UNCERTAIN_HEADS = frozenset(
    {AttentionHead(0, 10), AttentionHead(5, 8), AttentionHead(5, 9)}
)

# Legend order of the figures.
IOI_ROLE_ORDER = [
    NAME_MOVER,
    BACKUP_NAME_MOVER,
    NEGATIVE_NAME_MOVER,
    S_INHIBITION,
    DUPLICATE_TOKEN,
    INDUCTION,
    PREVIOUS_TOKEN,
    OTHER,
]

# Okabe-Ito colour-blind-safe palette, one colour per role.
IOI_ROLE_COLORS = {
    NAME_MOVER: "#0072B2",
    BACKUP_NAME_MOVER: "#009E73",
    NEGATIVE_NAME_MOVER: "#D55E00",
    S_INHIBITION: "#E69F00",
    DUPLICATE_TOKEN: "#CC79A7",
    INDUCTION: "#56B4E9",
    PREVIOUS_TOKEN: "#8B6914",
    OTHER: "#AAAAAA",
}

# Labels of the roles in the paper's figures and tables.
IOI_ROLE_ABBREVIATIONS = {
    NAME_MOVER: "NMH",
    BACKUP_NAME_MOVER: "BNMH",
    NEGATIVE_NAME_MOVER: "NNMH",
    S_INHIBITION: "S-Inh",
    DUPLICATE_TOKEN: "DTH",
    INDUCTION: "Induction",
    PREVIOUS_TOKEN: "PTH",
    OTHER: "other",
}


def ioi_role(head: AttentionHead) -> str:
    """Role of an attention head of GPT-2 small in the IOI circuit, ``OTHER`` outside it."""
    return IOI_HEAD_ROLES.get(head, OTHER)


def ioi_circuit_heads() -> list[AttentionHead]:
    """The 26 heads of the IOI circuit, sorted by layer and head."""
    return sorted(IOI_HEAD_ROLES)


def ioi_heads_with_roles(*roles: str) -> list[AttentionHead]:
    """Circuit heads that have one of ``roles``, grouped by role in the order given and,
    within a role, in the order of :data:`IOI_HEAD_ROLES`, which is the column order of the
    paper's tables."""
    return [
        head
        for role in roles
        for head, head_role in IOI_HEAD_ROLES.items()
        if head_role == role
    ]


# Head groups whose cross-interactions Section 5.3 and Appendix B report, in table order.
NAME_MOVER_GROUP = ioi_heads_with_roles(NAME_MOVER)
S_INHIBITION_HEADS = ioi_heads_with_roles(S_INHIBITION)
S_INHIBITION_PARTNER_HEADS = ioi_heads_with_roles(
    DUPLICATE_TOKEN, INDUCTION, PREVIOUS_TOKEN, NEGATIVE_NAME_MOVER
)
BACKUP_NAME_MOVER_HEADS = ioi_heads_with_roles(BACKUP_NAME_MOVER)
# The other 141 heads of GPT-2 small (12 layers of 12 heads), in layer and head order.
NON_NAME_MOVER_HEADS = [
    AttentionHead(layer, head)
    for layer in range(12)
    for head in range(12)
    if AttentionHead(layer, head) not in NAME_MOVER_GROUP
]


GREATER_THAN_EARLY_MLPS = [MlpLayer(layer) for layer in range(4)]
GREATER_THAN_LATE_MLPS = [MlpLayer(layer) for layer in range(8, 12)]

# Order of the mediators in the Greater-Than cross-interaction arrays.
GREATER_THAN_MLPS = GREATER_THAN_EARLY_MLPS + GREATER_THAN_LATE_MLPS
