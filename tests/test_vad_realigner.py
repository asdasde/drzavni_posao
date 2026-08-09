import unittest

from drzavni_posao.vad_realigner import (
    Segment,
    VadRealigner,
    VadRegion,
    VadSegment,
)


class SegmentTests(unittest.TestCase):
    def test_intersection_overlap_and_padding(self):
        segment = Segment(1.0, 4.0)

        self.assertEqual(segment.intersection(Segment(3.0, 5.0)), Segment(3.0, 4.0))
        self.assertEqual(segment.overlap_duration(Segment(3.0, 5.0)), 1.0)
        self.assertEqual(segment.overlap_duration(Segment(4.0, 5.0)), 0.0)
        self.assertEqual(segment.padded(0.1), Segment(0.9, 4.1))


class VadSegmentTests(unittest.TestCase):
    def test_natural_order_is_start_then_end(self):
        segments = [
            VadSegment(2.0, 3.0),
            VadSegment(1.0, 4.0),
            VadSegment(1.0, 2.0),
        ]

        self.assertEqual(
            sorted(segments),
            [
                VadSegment(1.0, 2.0),
                VadSegment(1.0, 4.0),
                VadSegment(2.0, 3.0),
            ],
        )
        self.assertEqual(segments[0].sort_key, (2.0, 3.0))

    def test_intersection_returns_overlapping_segment(self):
        overlap = VadSegment(1.0, 4.0).intersection(VadSegment(3.0, 5.0))

        self.assertEqual(overlap, VadSegment(3.0, 4.0))
        self.assertEqual(overlap.duration, 1.0)

    def test_intersection_returns_none_when_only_boundaries_touch(self):
        self.assertIsNone(VadSegment(1.0, 2.0).intersection(VadSegment(2.0, 3.0)))


class VadRegionTests(unittest.TestCase):
    def test_sorts_segments_and_uses_natural_order_key(self):
        region = VadRegion(
            (
                VadSegment(2.0, 3.0),
                VadSegment(0.0, 1.0),
            )
        )

        self.assertEqual(
            region.segments,
            (
                VadSegment(0.0, 1.0),
                VadSegment(2.0, 3.0),
            ),
        )
        self.assertEqual(region.sort_key, (0.0, 3.0))

    def test_intersection_preserves_gaps_between_vad_segments(self):
        region = VadRegion(
            (
                VadSegment(0.0, 1.0),
                VadSegment(2.0, 3.0),
            )
        )

        overlap = region.intersection(Segment(0.5, 2.5))

        self.assertEqual(
            overlap,
            VadRegion(
                (
                    VadSegment(0.5, 1.0),
                    VadSegment(2.0, 2.5),
                )
            ),
        )
        self.assertEqual(overlap.duration, 1.0)


class VadRealignerTests(unittest.TestCase):
    def test_prepares_split_segments_for_second_alignment(self):
        segment = {
            "start": 10.2,
            "end": 20.0,
            "text": "ne može tako. Đe ste ljudi",
            "avg_logprob": -0.2,
            "words": [
                {"word": "ne", "start": 10.2, "end": 10.5},
                {"word": "može", "start": 10.6, "end": 11.0},
                {"word": "tako.", "start": 12.0, "end": 16.0},
                {"word": "Đe", "start": 14.0, "end": 18.0},
                {"word": "ste", "start": 18.1, "end": 18.5},
                {"word": "ljudi", "start": 19.0, "end": 20.0},
            ],
        }
        realigner = VadRealigner.from_segments(
            [
                VadSegment(10.0, 13.0),
                VadSegment(17.0, 20.0),
            ],
            min_silence=1.0,
            padding=0.1,
        )

        untouched, realignment_inputs = realigner.prepare([segment])

        self.assertEqual(untouched, [])
        self.assertEqual(
            realignment_inputs,
            [
                {
                    "start": 10.1,
                    "end": 13.1,
                    "text": "ne može tako.",
                    "avg_logprob": -0.2,
                },
                {
                    "start": 16.9,
                    "end": 20.1,
                    "text": "Đe ste ljudi",
                    "avg_logprob": -0.2,
                },
            ],
        )

    def test_keeps_segment_when_word_assignment_is_ambiguous(self):
        segment = {
            "start": 10.0,
            "end": 20.0,
            "text": "stretched",
            "words": [
                {"word": "stretched", "start": 12.0, "end": 18.0},
            ],
        }
        realigner = VadRealigner.from_segments(
            [
                VadSegment(10.0, 13.0),
                VadSegment(17.0, 20.0),
            ],
            min_silence=1.0,
            padding=0.1,
        )

        untouched, realignment_inputs = realigner.prepare([segment])

        self.assertEqual(untouched, [segment])
        self.assertEqual(realignment_inputs, [])


if __name__ == "__main__":
    unittest.main()
