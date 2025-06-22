"""Test methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from miditoolkit import Instrument, MidiFile, Pedal
from torch import (
    FloatTensor as ptFloatTensor,
)
from torch import (
    IntTensor as ptIntTensor,
)
from torch import (
    Tensor as ptTensor,
)

import miditok
from miditok.utils.utils import miditoolkit_to_symusic

from .utils_tests import HERE, MIDI_PATHS_ALL

if TYPE_CHECKING:
    from pathlib import Path

    import miditoolkit
    import symusic
    from symusic import Score, Track


def test_convert_tensors() -> None:
    original = [[2, 6, 95, 130, 25, 15]]
    types = [ptTensor, ptIntTensor, ptFloatTensor]

    tokenizer = miditok.TSD()
    for type_ in types:
        tensor = type_(original)
        tokenizer(tensor)  # to make sure it passes as decorator
        as_list = miditok.midi_tokenizer.convert_ids_tensors_to_list(tensor)
        assert as_list == original


def test_tokenize_datasets_file_tree(tmp_path: Path) -> None:
    midi_paths = MIDI_PATHS_ALL

    # Check the file tree is copied
    tokenizer = miditok.TSD(miditok.TokenizerConfig())
    tokenizer.tokenize_dataset(midi_paths, tmp_path, overwrite_mode=True)
    json_paths = list(tmp_path.glob("**/*.json"))
    json_paths.sort(key=lambda x: x.stem)
    midi_paths.sort(key=lambda x: x.stem)
    for json_path, midi_path in zip(json_paths, midi_paths):
        assert json_path.relative_to(tmp_path).with_suffix(
            ".test"
        ) == midi_path.relative_to(HERE).with_suffix(".test"), (
            f"The file tree has not been reproduced as it should, for the file"
            f"{midi_path} tokenized {json_path}"
        )

    # Just make sure the non-overwrite mode doesn't crash
    tokenizer.tokenize_dataset(midi_paths, tmp_path, overwrite_mode=False)


def are_notes_equals(note1: miditoolkit.Note, note2: symusic.Note) -> bool:
    attrs_to_check = ("start", "pitch", "velocity", "end")
    for attr_to_check in attrs_to_check:
        if getattr(note1, attr_to_check) != getattr(note2, attr_to_check):
            return False
    return True


def are_control_changes_equals(
    cc1: miditoolkit.ControlChange, cc2: symusic.ControlChange
) -> bool:
    return cc1.time == cc2.time and cc1.number == cc2.number and cc1.value == cc2.value


def are_pitch_bends_equals(pb1: miditoolkit.PitchBend, pb2: symusic.PitchBend) -> bool:
    return pb1.time == pb2.time and pb1.pitch == pb2.value


def are_pedals_equals(sp1: miditoolkit.Pedal, sp2: symusic.Pedal) -> bool:
    return sp1.start == sp2.time and sp1.end == sp2.time + sp2.duration


def are_tracks_equals(track1: Instrument, track2: Track) -> int:
    err = 0
    for attr_ in ("program", "is_drum"):
        if getattr(track1, attr_) != getattr(track2, attr_):
            err += 1
    track1.notes.sort(key=lambda x: (x.start, x.pitch, x.end, x.velocity))
    track2.notes.sort(key=lambda x: (x.time, x.pitch, x.end, x.velocity))
    for note1, note2 in zip(track1.notes, track2.notes):
        if not are_notes_equals(note1, note2):
            err += 1
    for cc1, cc2 in zip(track1.control_changes, track2.controls):
        if not are_control_changes_equals(cc1, cc2):
            err += 1
    for pb1, pb2 in zip(track1.pitch_bends, track2.pitch_bends):
        if not are_pitch_bends_equals(pb1, pb2):
            err += 1
    # get pedals from the miditoolkit track
    if len(track1.control_changes) >= 2:
        # last_pedal_on_time is None when no Pedal control change is "on", and is equal
        # to the oldest Pedal control change time (tick) while no CC to end it has been
        # found. We first need to sort the CC messages
        track1.control_changes.sort(key=lambda cc: cc.time)
        last_pedal_on_time = None
        pedals_track1: list[miditoolkit.Pedal] = []
        for control_change in track1.control_changes:
            if control_change.number != 64:
                continue
            if last_pedal_on_time is not None and control_change.value < 64:
                pedals_track1.append(Pedal(last_pedal_on_time, control_change.time))
                last_pedal_on_time = None
            elif last_pedal_on_time is None and control_change.value >= 64:
                last_pedal_on_time = control_change.time
        for sp1, sp2 in zip(pedals_track1, track2.pedals):
            if not are_pedals_equals(sp1, sp2):
                err += 1

    return err


def are_tempos_equals(
    tempo_change1: miditoolkit.TempoChange, tempo_change2: symusic.Tempo
) -> bool:
    return tempo_change1.time == tempo_change2.time and round(
        tempo_change1.tempo, 3
    ) == round(tempo_change2.tempo, 3)


def are_time_signatures_equals(
    time_sig1: miditoolkit.TimeSignature, time_sig2: symusic.TimeSignature
) -> bool:
    return (
        time_sig1.time == time_sig2.time
        and time_sig1.numerator == time_sig2.numerator
        and time_sig1.denominator == time_sig2.denominator
    )


def are_key_signatures_equals(
    key_sig1: miditoolkit.KeySignature, key_sig2: symusic.KeySignature
) -> bool:
    # if key_sig1.time != key_sig2.time or key_sig1.key_number != key_sig2.key:
    # we don't test key signatures as they are decoded differently
    return key_sig1.time == key_sig2.time


def are_lyrics_or_markers_equals(
    lyric1: miditoolkit.Lyric, lyric2: symusic.core.TextMetaTick
) -> bool:
    return lyric1.time == lyric2.time and lyric1.text == lyric2.text


def are_midis_equals(midi_mtk: MidiFile, midi_sms: Score) -> bool:
    err = 0

    assert midi_mtk.ticks_per_beat == midi_sms.ticks_per_quarter
    if len(midi_mtk.tempo_changes) != len(midi_sms.tempos):
        num_same_consecutive_tempos = 0
        previous_tempo = midi_sms.tempos[0].tempo
        for tempo in midi_sms.tempos[1:]:
            if tempo.tempo == previous_tempo:
                num_same_consecutive_tempos += 1
            previous_tempo = tempo.tempo
        if len(midi_mtk.tempo_changes) + num_same_consecutive_tempos != len(
            midi_sms.tempos
        ):
            print(
                f"expected {len(midi_mtk.tempo_changes)} tempos,"
                f"got {len(midi_sms.tempos)}"
            )
            err += abs(len(midi_mtk.tempo_changes) - len(midi_sms.tempos))
    else:
        for tempo1, tempo2 in zip(midi_mtk.tempo_changes, midi_sms.tempos):
            if not are_tempos_equals(tempo1, tempo2):
                err += 1
    if len(midi_mtk.time_signature_changes) != len(midi_sms.time_signatures):
        print(
            f"expected {len(midi_mtk.time_signature_changes)}"
            f"time signatures, got {len(midi_sms.time_signatures)}"
        )
        err += abs(len(midi_mtk.time_signature_changes) - len(midi_sms.time_signatures))
    else:
        for ts1, ts2 in zip(midi_mtk.time_signature_changes, midi_sms.time_signatures):
            if not are_time_signatures_equals(ts1, ts2):
                err += 1
    # Not testing lyrics anymore as symusic contain them at the track level
    """if len(midi_mtk.lyrics) != len(midi_sms.lyrics):
        print(f"expected {len(midi_mtk.lyrics)} lyrics, got {len(midi_sms.lyrics)}")
        err += abs(len(midi_mtk.lyrics) - len(midi_sms.lyrics))
    else:
        for lyrics1, lyrics2 in zip(midi_mtk.lyrics, midi_sms.lyrics):
            if not are_lyrics_or_markers_equals(lyrics1, lyrics2):
                err += 1"""
    if len(midi_mtk.markers) != len(midi_sms.markers):
        print(f"expected {len(midi_mtk.markers)} markers, got {len(midi_sms.markers)}")
        err += abs(len(midi_mtk.markers) - len(midi_sms.markers))
    else:
        for marker1, marker2 in zip(midi_mtk.markers, midi_sms.markers):
            if not are_lyrics_or_markers_equals(marker1, marker2):
                err += 1

    # Check tracks: notes, control changes, pitch bends
    midi_mtk.instruments.sort(key=lambda t: (t.program, t.is_drum, len(t.notes)))
    midi_sms.tracks.sort(key=lambda t: (t.program, t.is_drum, len(t.notes)))
    for track1, track2 in zip(midi_mtk.instruments, midi_sms.tracks):
        err += are_tracks_equals(track1, track2)

    return err == 0


def test_miditoolkit_to_symusic() -> None:
    midi = MidiFile(MIDI_PATHS_ALL[0])
    score = miditoolkit_to_symusic(midi)

    assert are_midis_equals(midi, score)


def test_legacy_miditoolkit() -> None:
    midi = MidiFile(MIDI_PATHS_ALL[0])
    tokenizer = miditok.TSD()
    _ = tokenizer(midi)
