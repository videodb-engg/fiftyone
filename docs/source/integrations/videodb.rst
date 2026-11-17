.. _videodb-integration:

VideoDB Integration
===================

.. default-role:: code

FiftyOne integrates with `VideoDB <https://videodb.io>`_, whose video
understanding analyzers let you add timestamped transcripts and scene
descriptions to your video datasets with a few lines of code:

-   **Transcripts** split the speech in each video into timed segments, with
    word-level timing and speaker labels
-   **Scene descriptions** split each video into scenes and describe what
    happens in each one, optionally following your own prompt and output
    schema

Both are stored as
:class:`TemporalDetections <fiftyone.core.labels.TemporalDetections>`, so you
can filter them, browse them on the App's timeline, and turn them into clips.
The analyzers run server-side via the VideoDB API, so no local GPU is
required.

.. _videodb-setup:

Setup
_____

Install the `videodb` package:

.. code-block:: shell
    :linenos:

    pip install "videodb>=0.5.1"

You can grab a free API key at
`console.videodb.io <https://console.videodb.io>`_. Provide it via the
``VIDEO_DB_API_KEY`` environment variable:

.. code-block:: shell
    :linenos:

    export VIDEO_DB_API_KEY=...

or pass it directly via the ``api_key`` config parameter when loading the
model.

.. note::

    Each video is uploaded to your VideoDB collection before it is analyzed.
    Pass ``collection_id=...`` to use a collection other than your default
    one.

.. _videodb-transcripts:

Video transcripts
_________________

Transcribe the speech in your videos into timed segments:

.. code-block:: python
    :linenos:

    import fiftyone as fo
    import fiftyone.zoo as foz
    from fiftyone.utils.videodb import VideoDBModel, VideoDBModelConfig

    dataset = foz.load_zoo_dataset("quickstart-video")

    # Load directly
    model = VideoDBModel(VideoDBModelConfig({"operation": "transcribe"}))

    # Load via zoo
    # model = foz.load_zoo_model("videodb-transcript")

    dataset.apply_model(model, label_field="transcript")

    session = fo.launch_app(dataset)

Each transcript segment is a
:class:`TemporalDetection <fiftyone.core.labels.TemporalDetection>` with label
``"transcript"``, whose ``text`` attribute holds the spoken text and whose
``words`` attribute holds the timing, confidence, and speaker of each word.

You can pass a language hint for non-English videos:

.. code-block:: python
    :linenos:

    model = foz.load_zoo_model("videodb-transcript", language="hi")

.. _videodb-scenes:

Scene descriptions
__________________

Split your videos into scenes and describe each one:

.. code-block:: python
    :linenos:

    import fiftyone as fo
    import fiftyone.zoo as foz
    from fiftyone.utils.videodb import VideoDBModel, VideoDBModelConfig

    dataset = foz.load_zoo_dataset("quickstart-video")

    # Load directly
    model = VideoDBModel(VideoDBModelConfig({"operation": "describe"}))

    # Load via zoo
    # model = foz.load_zoo_model("videodb-scene-descriptions")

    dataset.apply_model(model, label_field="scenes")

    session = fo.launch_app(dataset)

Each scene is a
:class:`TemporalDetection <fiftyone.core.labels.TemporalDetection>` with label
``"scene"`` whose ``text`` attribute holds its description.

By default, scenes are split at visual cuts. You can split them into fixed
windows instead, and provide your own prompt and an output ``schema`` whose
fields are stored as attributes of each scene:

.. code-block:: python
    :linenos:

    from fiftyone import ViewField as F

    model = foz.load_zoo_model(
        "videodb-scene-descriptions",
        prompt="Describe the activity and the setting of this scene.",
        schema={"activity": "string", "setting": "string"},
        segmentation={"type": "time", "seconds": 5},
        model="pro",  # "mini", "basic", "pro", or "ultra"
    )

    dataset.apply_model(model, label_field="activities")

    # Find all outdoor scenes
    view = dataset.filter_labels(
        "activities", F("setting").contains_str("outdoor")
    )

    # Turn each matching scene into its own clip
    clips = view.to_clips("activities")

    session = fo.launch_app(clips)
