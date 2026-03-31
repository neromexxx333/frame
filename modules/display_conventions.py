"""
Aturan tampilan tanda gaya/momen untuk tabel dan diagram.
"""

from typing import Optional, Tuple


# Permintaan tampilan khusus: momen joint balok pada node 5
# ditampilkan mengikuti tanda aksi joint solver agar pembacaan
# E7@node5 dan E8@node5 konsisten dengan cek node.
SPECIAL_BEAM_JOINT_RAW_SIGN_NODE_IDS = {5}


def normalize_element_code(code: str) -> str:
    """Normalkan kode elemen menjadi huruf kapital tunggal."""
    return str(code or "").strip().upper()


def use_raw_beam_joint_moment_display(code: str,
                                      joint_node_id: Optional[int]) -> bool:
    """True bila momen joint balok di node tertentu ditampilkan tanpa dibalik."""
    if normalize_element_code(code) != 'B' or joint_node_id is None:
        return False
    try:
        return int(joint_node_id) in SPECIAL_BEAM_JOINT_RAW_SIGN_NODE_IDS
    except (TypeError, ValueError):
        return False


def get_displayed_joint_moment(raw_value: float,
                               code: str,
                               joint_node_id: Optional[int]) -> float:
    """Konversi momen joint solver menjadi tanda yang ditampilkan di UI."""
    raw_value = float(raw_value)
    if normalize_element_code(code) != 'B':
        return raw_value
    if use_raw_beam_joint_moment_display(code, joint_node_id):
        return raw_value
    return -raw_value


def get_displayed_end_internal_moment(raw_value: float,
                                      code: str,
                                      joint_node_id: Optional[int]) -> float:
    """Konversi momen internal tepat sebelum ujung elemen untuk UI."""
    return get_displayed_joint_moment(raw_value, code, joint_node_id)


def get_displayed_element_moment_values(code: str,
                                        raw_start_joint: float,
                                        raw_end_joint: float,
                                        raw_end_internal: float,
                                        node_start: Optional[int],
                                        node_end: Optional[int]) -> Tuple[float, float, float]:
    """Kembalikan pasangan momen tampil Start/End_Joint/End_Internal."""
    return (
        get_displayed_joint_moment(raw_start_joint, code, node_start),
        get_displayed_joint_moment(raw_end_joint, code, node_end),
        get_displayed_end_internal_moment(raw_end_internal, code, node_end)
    )


def get_joint_equilibrium_moment(display_value: float,
                                 code: str,
                                 joint_node_id: Optional[int]) -> float:
    """Ubah nilai tabel menjadi kontribusi momen joint untuk cek node."""
    display_value = float(display_value)
    if normalize_element_code(code) != 'B':
        return display_value
    if use_raw_beam_joint_moment_display(code, joint_node_id):
        return display_value
    return -display_value


def use_raw_beam_moment_profile_for_plot(code: str,
                                         node_start: Optional[int],
                                         node_end: Optional[int]) -> bool:
    """
    Tentukan orientasi profil momen balok pada plot.

    Bila ujung start balok memakai tanda joint asli solver sementara ujung
    lainnya tetap memakai konvensi tabel standar, profil raw dipakai agar
    warna dan label di plot tetap selaras dengan tanda tampilan baru.
    """
    if normalize_element_code(code) != 'B':
        return False
    return (
        use_raw_beam_joint_moment_display(code, node_start)
        and not use_raw_beam_joint_moment_display(code, node_end)
    )
