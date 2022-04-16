import gdspy
import numpy as np
from skimage.draw import rectangle


def get_srafs_vias_from_gds(gds_path):
    gdsii = gdspy.GdsLibrary(infile=gds_path)
    layers = gdsii.cells['TOP_new'].get_polygons(by_spec=True)
    srafs = layers[2, 0]
    vias = layers[0, 0]
    return srafs, vias

def get_layout_location(layout):
    co = np.array(layout).reshape(-1, 2)
    x_min, y_min = np.min(co, axis=0)
    x_max, y_max = np.max(co, axis=0)
    return x_min, y_min, x_max, y_max

def to_grid(co, offset, step):
    return np.uint32(np.floor((co + offset) / step))

def shape_to_grid(shape, x_offset, y_offset, step):
    x1, y1, x2, y2 = get_layout_location(shape)
    x1 = to_grid(x1, x_offset, step)
    x2 = to_grid(x2, x_offset, step)
    y1 = to_grid(y1, y_offset, step)
    y2 = to_grid(y2, y_offset, step)
    return x1, x2, y1, y2

def gen_shapes(shapes, out_shape, x_offset, y_offset, step):
    gen_img = np.zeros(out_shape, dtype=np.uint8)
    for v in shapes:
        x1, x2, y1, y2 = shape_to_grid(v, x_offset, y_offset, step)
        rec = rectangle((y1, x1), end=(y2, x2))
        gen_img[tuple(rec)] = 255
    return gen_img

