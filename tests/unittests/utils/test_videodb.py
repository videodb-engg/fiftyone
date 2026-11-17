"""
Tests for the fiftyone/utils/videodb.py VideoDB integration.

The tests use a mocked VideoDB connection, so no API key or network access is
required.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from unittest import mock

import pytest

import fiftyone as fo  # pylint: disable=unused-import
import fiftyone.core.labels as fol
import fiftyone.core.metadata as fome
import fiftyone.utils.videodb as fouv

pytest.importorskip("videodb")

_VIDEO_URL = "https://example.com/video.mp4"

_METADATA = fome.VideoMetadata(
    total_frame_count=300, frame_rate=30.0, duration=10.0
)


def _config(**kwargs):
    return fouv.VideoDBModelConfig(kwargs)


def _analyzer(status="done", output=None):
    analyzer = mock.MagicMock()
    analyzer.status = status
    analyzer.is_successful = status == "done"
    analyzer.get_output.return_value = output
    return analyzer


@pytest.fixture
def video():
    video = mock.MagicMock()
    video.understand.return_value.id = "understanding-1"
    return video


@pytest.fixture
def conn(video):
    conn = mock.MagicMock()
    conn.get_collection.return_value.upload.return_value = video
    return conn


def _model(conn, **kwargs):
    kwargs.setdefault("api_key", "test-key")
    model = fouv.VideoDBModel(_config(**kwargs))
    model._conn = conn
    return model


def _predict(model, path=_VIDEO_URL):
    with mock.patch.object(
        model, "_get_video_metadata", return_value=_METADATA
    ):
        return model.predict(path)


class TestVideoDBModelConfig:
    def test_defaults(self):
        config = _config()

        assert config.operation == "transcribe"
        assert config.api_key is None
        assert config.collection_id is None
        assert config.prompt is None
        assert config.schema is None
        assert config.segmentation is None
        assert config.timeout == 1800
        assert config.poll_interval == 10

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"operation": "embed"}, "Unsupported operation"),
            ({"prompt": "What?"}, "prompt is not supported"),
            ({"model": "pro"}, "model is not supported"),
            (
                {"segmentation": {"type": "time", "seconds": 5}},
                "segmentation is not supported",
            ),
            (
                {"operation": "describe", "language": "en"},
                "language is not supported",
            ),
        ],
    )
    def test_invalid_values_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _config(**kwargs)


class TestVideoDBModel:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("VIDEO_DB_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No VideoDB API key"):
            fouv.VideoDBModel(_config())

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("VIDEO_DB_API_KEY", "env-key")
        with mock.patch.object(fouv.videodb, "connect") as connect:
            fouv.VideoDBModel(_config())

        connect.assert_called_once_with(api_key="env-key")

    def test_transcribe(self, conn, video):
        output = {
            "scenes": [
                {
                    "start": 0.0,
                    "end": 4.0,
                    "data": {
                        "text": "Hello there.",
                        "words": [{"text": "Hello", "speaker": "A"}],
                    },
                },
                {"start": 5.0, "end": 10.0, "data": {"text": "Goodbye."}},
            ]
        }
        analyzer = _analyzer(output=output)
        video.understand.return_value.get_analyzer.return_value = analyzer
        model = _model(conn, language="en")

        label = _predict(model)

        assert isinstance(label, fol.TemporalDetections)
        first, second = label.detections
        assert first.label == "transcript"
        assert first.support == [1, 121]
        assert first["text"] == "Hello there."
        assert first["words"][0]["speaker"] == "A"
        assert second.support == [151, 300]

        conn.get_collection.assert_called_once_with("default")
        upload = conn.get_collection.return_value.upload
        upload.assert_called_once_with(source=_VIDEO_URL)

        kwargs = video.understand.call_args.kwargs
        assert kwargs["analyzers"] == [
            {
                "type": "spoken_words",
                "name": "transcript",
                "config": {"language": "en"},
            }
        ]
        assert kwargs["segmentation"] is None

        understanding = video.understand.return_value
        understanding.wait_until_complete.assert_called_once_with(
            timeout=1800, poll_interval=10
        )
        understanding.get_analyzer.assert_called_once_with(
            "transcript", refresh=True
        )

    def test_describe(self, conn, video):
        output = {
            "scenes": [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "data": {"activity": "running", "label": "reserved"},
                }
            ]
        }
        analyzer = _analyzer(output=output)
        video.understand.return_value.get_analyzer.return_value = analyzer
        model = _model(
            conn,
            operation="describe",
            collection_id="c-123",
            prompt="Describe the activity.",
            schema={"activity": "string"},
            model="pro",
            segmentation={"type": "time", "seconds": 5},
            timeout=60,
            poll_interval=1,
        )

        label = _predict(model)

        (detection,) = label.detections
        assert detection.label == "scene"
        assert detection.support == [1, 300]
        assert detection["activity"] == "running"
        assert detection["label_"] == "reserved"

        conn.get_collection.assert_called_once_with("c-123")

        kwargs = video.understand.call_args.kwargs
        assert kwargs["analyzers"] == [
            {
                "type": "vlm",
                "name": "scene",
                "config": {
                    "prompt": "Describe the activity.",
                    "schema": {"activity": "string"},
                    "model": "pro",
                },
            }
        ]
        assert kwargs["segmentation"] == {"type": "time", "seconds": 5}

        understanding = video.understand.return_value
        understanding.wait_until_complete.assert_called_once_with(
            timeout=60, poll_interval=1
        )

    def test_default_analyzer_has_no_config(self, conn, video):
        analyzer = _analyzer(output={"scenes": []})
        video.understand.return_value.get_analyzer.return_value = analyzer
        model = _model(conn, operation="describe")

        label = _predict(model)

        assert label.detections == []
        kwargs = video.understand.call_args.kwargs
        assert kwargs["analyzers"] == [{"type": "vlm", "name": "scene"}]

    def test_video_is_uploaded_once(self, conn, video):
        analyzer = _analyzer(output={"scenes": []})
        video.understand.return_value.get_analyzer.return_value = analyzer
        model = _model(conn)

        _predict(model)
        _predict(model)

        conn.get_collection.assert_called_once()
        conn.get_collection.return_value.upload.assert_called_once()
        assert video.understand.call_count == 2

    def test_failed_analyzer_raises(self, conn, video):
        analyzer = _analyzer(status="failed")
        video.understand.return_value.get_analyzer.return_value = analyzer
        model = _model(conn)

        with pytest.raises(RuntimeError, match="status 'failed'"):
            _predict(model)

        analyzer.get_output.assert_not_called()

    def test_unsupported_input_raises(self, conn):
        model = _model(conn)

        with pytest.raises(ValueError, match="Unsupported input"):
            model.predict(123)

    def test_video_metadata_from_reader(self):
        reader = mock.MagicMock(spec=fouv.etav.FFmpegVideoReader)
        reader.total_frame_count = 300
        reader.frame_rate = 30.0

        metadata = fouv.VideoDBModel._get_video_metadata(reader, "video.mp4")

        assert metadata.total_frame_count == 300
        assert metadata.duration == 10.0
