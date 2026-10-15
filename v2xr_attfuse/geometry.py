"""Check existing IoU/NMS inputs and outputs while preserving their calculations."""
class CheckedGeometry:
    def __init__(self, common_utils, np):
        self.common, self.np = common_utils, np
        self.original_convert = common_utils.convert_format
        self.original_iou = common_utils.compute_iou
        self.polygons = self.pairs = 0

    def convert(self, boxes):
        if not self.np.isfinite(boxes).all():
            raise ValueError('Nonfinite decoded box coordinates')
        values = self.original_convert(boxes)
        for polygon in values:
            if (not polygon.is_valid or polygon.is_empty or not self.np.isfinite(polygon.area)
                    or polygon.area <= 0):
                from shapely.validation import explain_validity
                raise ValueError(f'Invalid polygon: {explain_validity(polygon)}; {polygon.wkt}')
        self.polygons += len(values)
        return values

    def iou(self, box, boxes):
        values = self.original_iou(box, boxes)
        if not self.np.isfinite(values).all() or ((values < -1e-6) | (values > 1 + 1e-6)).any():
            raise ValueError(f'Invalid IoU values: {values!r}')
        self.pairs += len(values)
        return values

    def __enter__(self):
        self.common.convert_format, self.common.compute_iou = self.convert, self.iou
        return self

    def __exit__(self, *_):
        self.common.convert_format, self.common.compute_iou = self.original_convert, self.original_iou
