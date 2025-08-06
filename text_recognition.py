from paddleocr import TextRecognition


class TextRecognitionSingleton(object):
  def __new__(cls):
    if not hasattr(cls, 'instance'):
        cls.model = TextRecognition()
        cls.instance = super(TextRecognitionSingleton, cls).__new__(cls)
    return cls.instance