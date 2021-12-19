from funscript_editor.ui.opencvui import OpenCV_GUI, OpenCV_GUI_Parameters
from funscript_editor.data.ffmpegstream import FFmpegStream

test_video = './tests/data/example1.mkv'

params = OpenCV_GUI_Parameters(
        video_info=FFmpegStream.get_video_info(test_video)
    )

ui = OpenCV_GUI(params)

first_frame = FFmpegStream.get_frame(test_video, 0)
projection_config = ui.get_video_projection_config(first_frame, 'vr_he_180_sbs')
video = FFmpegStream(
        video_path = test_video,
        config = projection_config,
        skip_frames = 1,
        start_frame = 0
    )

first_frame = video.read()
box = ui.bbox_selector(first_frame, "Test box selector", use_zoom=False)
print(box)

next_frame = video.read()
selection = ui.min_max_selector(first_frame, next_frame, "Test Min Max selector", "Min", "Max")
print(selection)
