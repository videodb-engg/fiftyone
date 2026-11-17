"""
`VideoDB <https://videodb.io>`_ video understanding integration.

This module provides a :class:`fiftyone.core.models.Model` that runs
`VideoDB <https://docs.videodb.io>`_ understanding analyzers on your videos
and stores their timestamped outputs as
:class:`fiftyone.core.labels.TemporalDetections`:

-   **Transcripts** split the speech in each video into timed segments
-   **Scene descriptions** describe what happens in each scene of a video,
    optionally following your own prompt and output schema

The analyzers run server-side via the VideoDB API, so no local GPU is
required. Set your API key via the ``VIDEO_DB_API_KEY`` environment variable
or pass ``api_key=...`` when loading the model.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import logging
import os

import eta.core.video as etav

import fiftyone.core.config as foc
import fiftyone.core.labels as fol
import fiftyone.core.metadata as fome
import fiftyone.core.models as fom
import fiftyone.core.utils as fou
import fiftyone.zoo.models as fozm

videodb = fou.lazy_import(
    "videodb", callback=lambda: fou.ensure_package("videodb>=0.5.1")
)

logger = logging.getLogger(__name__)

# Maps each operation to its VideoDB analyzer type and artifact name
_ANALYZERS = {
    "transcribe": ("spoken_words", "transcript"),
    "describe": ("vlm", "scene"),
}

# Label fields that analyzer outputs must not overwrite
_RESERVED_ATTRS = {
    "id",
    "_id",
    "_cls",
    "label",
    "support",
    "confidence",
    "tags",
    "attributes",
}


class VideoDBModelConfig(foc.Config, fozm.HasZooModel):
    """Configuration for running a :class:`VideoDBModel`.

    Args:
        operation ("transcribe"): the operation to perform when the model is
            applied. Supported values are ``"transcribe"`` (timed transcript
            segments) and ``"describe"`` (scene descriptions)
        api_key (None): the VideoDB API key to use. If not provided, the
            ``VIDEO_DB_API_KEY`` environment variable is used
        collection_id (None): the ID of the VideoDB collection to upload
            videos to. By default, your default collection is used
        language (None): an optional language code, such as ``"en"``, for
            ``"transcribe"`` operations
        prompt (None): an optional prompt for ``"describe"`` operations
        schema (None): an optional dict mapping field names to types, such
            as ``{"activity": "string"}``, that ``"describe"`` operations
            extract for each scene
        model (None): an optional VideoDB model tier for ``"describe"``
            operations, such as ``"mini"``, ``"basic"``, ``"pro"``, or
            ``"ultra"``
        segmentation (None): an optional dict that controls how videos are
            split into scenes for ``"describe"`` operations, such as
            ``{"type": "time", "seconds": 5}`` or
            ``{"type": "shot", "threshold": 30}``
        timeout (1800): the maximum time, in seconds, to wait for each video
            to be analyzed
        poll_interval (10): the time, in seconds, between status checks while
            waiting for each video to be analyzed
    """

    def __init__(self, d):
        self.operation = self.parse_string(
            d, "operation", default="transcribe"
        )
        if self.operation not in _ANALYZERS:
            raise ValueError(
                "Unsupported operation '%s'. Supported values are %s"
                % (self.operation, tuple(_ANALYZERS.keys()))
            )

        self.api_key = self.parse_string(d, "api_key", default=None)
        self.collection_id = self.parse_string(
            d, "collection_id", default=None
        )
        self.language = self.parse_string(d, "language", default=None)
        self.prompt = self.parse_string(d, "prompt", default=None)
        self.schema = self.parse_dict(d, "schema", default=None)
        self.model = self.parse_string(d, "model", default=None)
        self.segmentation = self.parse_dict(d, "segmentation", default=None)
        self.timeout = self.parse_number(d, "timeout", default=1800)
        self.poll_interval = self.parse_number(d, "poll_interval", default=10)

        if self.operation == "transcribe":
            options = ("prompt", "schema", "model", "segmentation")
        else:
            options = ("language",)

        for name in options:
            if getattr(self, name) is not None:
                raise ValueError(
                    "%s is not supported for '%s' operations"
                    % (name, self.operation)
                )


class VideoDBModel(fom.Model):
    """Wrapper for running `VideoDB <https://videodb.io>`_ video understanding
    analyzers.

    Each video is uploaded to your VideoDB collection once per model instance,
    and the analyzer's timestamped records are returned as
    :class:`fiftyone.core.labels.TemporalDetections`.

    Example usage::

        import fiftyone as fo
        import fiftyone.zoo as foz
        from fiftyone.utils.videodb import VideoDBModel, VideoDBModelConfig

        dataset = foz.load_zoo_dataset("quickstart-video", max_samples=2)

        #
        # Timed transcript segments
        #

        # Load directly
        model = VideoDBModel(VideoDBModelConfig({"operation": "transcribe"}))

        # Load via zoo
        # model = foz.load_zoo_model("videodb-transcript")

        dataset.apply_model(model, label_field="transcript")

        #
        # Scene descriptions
        #

        # Load directly
        model = VideoDBModel(VideoDBModelConfig({"operation": "describe"}))

        # Load via zoo
        # model = foz.load_zoo_model("videodb-scene-descriptions")

        dataset.apply_model(model, label_field="scenes")

        session = fo.launch_app(dataset)

    Args:
        config: a :class:`VideoDBModelConfig`
    """

    def __init__(self, config):
        self.config = config

        api_key = config.api_key or os.environ.get("VIDEO_DB_API_KEY")
        if not api_key:
            raise ValueError(
                "No VideoDB API key found. Provide one via the `api_key` "
                "config parameter or the `VIDEO_DB_API_KEY` environment "
                "variable. You can get a free key at https://console.videodb.io"
            )

        self._conn = videodb.connect(api_key=api_key)
        self._collection = None
        self._videos = {}

    @property
    def media_type(self):
        return "video"

    @property
    def ragged_batches(self):
        return False

    @property
    def transforms(self):
        return None

    @property
    def preprocess(self):
        return False

    @preprocess.setter
    def preprocess(self, value):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def predict(self, arg):
        """Analyzes the given video.

        Args:
            arg: an active ``eta.core.video.FFmpegVideoReader`` or a video
                filepath/URL

        Returns:
            a :class:`fiftyone.core.labels.TemporalDetections` with one
            :class:`fiftyone.core.labels.TemporalDetection` per transcript
            segment or scene
        """
        path = self._get_path(arg)
        video = self._get_video(path)
        output = self._understand(video)
        metadata = self._get_video_metadata(arg, path)

        _, name = _ANALYZERS[self.config.operation]
        return _parse_scenes(output, metadata, name)

    @staticmethod
    def _get_path(arg):
        if isinstance(arg, etav.FFmpegVideoReader):
            return arg.inpath

        if isinstance(arg, str):
            return arg

        raise ValueError(
            "Unsupported input of type %s; expected a video filepath or an "
            "`eta.core.video.FFmpegVideoReader`" % type(arg)
        )

    @staticmethod
    def _get_video_metadata(arg, path):
        if isinstance(arg, etav.FFmpegVideoReader):
            total_frame_count = arg.total_frame_count
            frame_rate = arg.frame_rate
            duration = total_frame_count / frame_rate if frame_rate else None
            return fome.VideoMetadata(
                total_frame_count=total_frame_count,
                frame_rate=frame_rate,
                duration=duration,
            )

        return fome.VideoMetadata.build_for(path)

    def _get_video(self, path):
        # Uploads each video once per path; upload() handles files and URLs
        video = self._videos.get(path, None)
        if video is not None:
            return video

        if self._collection is None:
            self._collection = self._conn.get_collection(
                self.config.collection_id or "default"
            )

        video = self._collection.upload(source=path)
        self._videos[path] = video
        return video

    def _understand(self, video):
        analyzer_type, name = _ANALYZERS[self.config.operation]

        config = {}
        if self.config.language is not None:
            config["language"] = self.config.language
        if self.config.prompt is not None:
            config["prompt"] = self.config.prompt
        if self.config.schema is not None:
            config["schema"] = self.config.schema
        if self.config.model is not None:
            config["model"] = self.config.model

        analyzer = {"type": analyzer_type, "name": name}
        if config:
            analyzer["config"] = config

        understanding = video.understand(
            analyzers=[analyzer], segmentation=self.config.segmentation
        )
        understanding.wait_until_complete(
            timeout=self.config.timeout,
            poll_interval=self.config.poll_interval,
        )

        analyzer = understanding.get_analyzer(name, refresh=True)
        if not analyzer.is_successful:
            raise RuntimeError(
                "VideoDB analyzer '%s' of understanding run '%s' finished "
                "with status '%s'" % (name, understanding.id, analyzer.status)
            )

        return analyzer.get_output()


def _parse_scenes(output, metadata, label):
    detections = []
    for scene in (output or {}).get("scenes") or []:
        detection = fol.TemporalDetection.from_timestamps(
            [scene["start"], scene["end"]], metadata=metadata, label=label
        )

        for name, value in (scene.get("data") or {}).items():
            if name in _RESERVED_ATTRS:
                name += "_"

            detection[name] = value

        detections.append(detection)

    return fol.TemporalDetections(detections=detections)
