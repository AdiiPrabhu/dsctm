from scripts.build_daicwoz_participant_egemaps88 import participant_intervals


def test_participant_intervals_exclude_interviewer_and_invalid_rows(tmp_path):
    transcript = tmp_path / "transcript.tsv"
    transcript.write_text(
        "start_time\tstop_time\tspeaker\tvalue\n"
        "1.0\t2.0\tEllie\thello\n"
        "2.5\t4.0\tParticipant\tanswer\n"
        "5.0\t4.0\tParticipant\tbad interval\n"
        "6.0\t7.0\t participant \tmore\n"
    )
    assert participant_intervals(transcript) == [(2.5, 4.0), (6.0, 7.0)]
