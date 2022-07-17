# Copyright (c) OpenMMLab. All rights reserved.
from .audio_recognizer import AudioRecognizer
from .base import BaseRecognizer
from .recognizer2d import Recognizer2D
from .recognizer3d import Recognizer3D
from .recognizer_tempagg import RecognizerTempAgg
from .recognizer_i3d_da import RecognizerI3dDA
__all__ = ['BaseRecognizer', 'Recognizer2D', 'Recognizer3D', 'AudioRecognizer', 'RecognizerTempAgg', 'RecognizerI3dDA']
