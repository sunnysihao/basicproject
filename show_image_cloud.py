# 安装：pip install numpy opencv-python matplotlib Pillow
# 使用：python show_image_cloud.py <pcd文件> <图片文件> <json配置文件>
import numpy as np
import re
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt


numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


def _build_dtype(metadata):
    """ Build numpy structured array dtype from pcl metadata.

    Note that fields with count > 1 are 'flattened' by creating multiple
    single-count fields.

    *TODO* allow 'proper' multi-count fields.
    """
    fieldnames = []
    typenames = []
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata.get('type', 'F'),
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            fieldnames.extend(['%s_%04d' % (f, i) for i in range(c)])
            typenames.extend([np_type]*c)
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype


def parse_header(lines):
    """ Parse header of PCD files.
    """
    metadata = {}
    for ln in lines:
        if ln.startswith('#') or len(ln) < 2:
            continue
        match = re.match('(\w+)\s+([\w\s\.]+)', ln)
        if not match:
            print("warning: can't understand line: %s" % ln)
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = list(map(int, value.split()))
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = map(float, value.split())
        elif key == 'data':
            metadata[key] = value.strip().lower()
        # TODO apparently count is not required?
    # add some reasonable defaults
    if 'count' not in metadata:
        metadata['count'] = [1]*len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata


def load_pcd(pcd_file):
    with open(pcd_file, 'rb') as f:
        header = []
        for _ in range(11):
            ln = f.readline().decode("ascii").strip()
            header.append(ln)
            if ln.startswith('DATA'):
                metadata = parse_header(header)
                dtype = _build_dtype(metadata)
                code = metadata['data']
                break
        else:
            raise ValueError("invalid file header")

        if code == 'ascii':
            lines = [line.strip() for line in f]
            rows = [
                [
                    float(field)
                    for field in line.split()
                ] 
                for line in lines if line
            ]
            pc = np.array(rows)
            
            # pc = np.loadtxt(f, dtype=dtype, delimiter=' ')
        elif code == 'binary':
            rowstep = metadata['points']*dtype.itemsize
            # for some reason pcl adds empty space at the end of files
            buf = f.read(rowstep)
            pc = np.fromstring(buf, dtype=dtype)
        # elif code == 'binary_compressed':
        #     pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
        else:
            # or "binary_compressed"
            raise ValueError('DATA field is neither "ascii" or "binary"')
    pc = np.asarray(pc.tolist())
    N, C = pc.shape
    if C == 3:
        pc = np.hstack([pc, np.ones((N, 1))])
    else:
        assert C > 3
        pc = pc[:, :4]
        pc[:, 3] = 1.0
    return pc


def get_2d_lidar_projection(pc: np.ndarray, intrinsic: np.ndarray,
                            extrinsic: np.ndarray = None) -> tuple:
    """

    :param pc: (N, 4)
    :param intrinsic: (4, 4)
    :param extrinsic: (4, 4)
    :return:
    """
    assert len(
        pc.shape) == 2 and pc.shape[1] == 4, f"pc shape must be (N, 4), not {pc.shape}"
    assert abs(pc[:10, 3].mean() - 1) < 1e-5, "pc intensity data must be 1.0"
    assert intrinsic.shape == (
        4, 4), f'intrinsic shape must be (4, 4), not {intrinsic.shape}'
    assert extrinsic is None or extrinsic.shape == (
        4, 4), f'extrinsic shape must be (4, 4), not {extrinsic.shape}'

    # apply extrinsic
    pc_xyz = pc if extrinsic is None else pc @ extrinsic.T

    # projection
    pc_xyz = pc_xyz @ intrinsic.T

    pc_z = pc_xyz[:, 2]
    pc_xyz = pc_xyz / (pc_xyz[:, 2, None] + 1e-10)
    pc_uv = pc_xyz[:, :2]
    return pc_uv, pc_z


def sort_points(pc_uv: np.ndarray, pc_z: np.ndarray) -> tuple:
    """sort points from far to near

    :param pc_uv:
    :param pc_z:
    :return:
    """
    assert len(pc_uv.shape) == 2 and pc_uv.shape[1] == 2
    assert len(pc_z.shape) == 1 or (
        len(pc_z.shape) == 2 and pc_z.shape[1] == 1)
    assert pc_uv.shape[0] == pc_z.shape[0]

    # sort by z descending
    indices = pc_z.flatten().argsort(axis=0)[::-1]
    return pc_uv[indices], pc_z[indices]


def fusion_images(foreground: np.ndarray, background: np.ndarray, alpha=0.5) -> np.ndarray:
    """

    :param foreground: (H, W, C)
    :param background: (H, W, C)
    :param alpha:
    :return:
    """
    image = cv2.addWeighted(foreground, alpha, background, 1 - alpha, 0)
    return image


def get_projected_pts(pc: np.ndarray, intrinsic: np.ndarray, img_shape: tuple,
                      extrinsic: np.ndarray = None, verbose=False) -> tuple:
    """
    :param pc: (N, 4)
    :param intrinsic: (4, 4)
    :param img_shape: (H, W, C)
    :param extrinsic: (4, 4)
    :param verbose:
    :return:
    """
    assert pc.shape[1] == 4
    H, W, C = img_shape
    assert H > 100 and W > 100 and C == 3
    pc_uv, pc_z = get_2d_lidar_projection(pc, intrinsic, extrinsic=extrinsic)
    mask = (pc_uv[:, 0] > 0) & (pc_uv[:, 0] < W) & (
        pc_uv[:, 1] > 0) & (pc_uv[:, 1] < H) & (pc_z > 0)
    uv, z = pc_uv[mask], pc_z[mask]
    if len(uv) == 0:
        uv, z = np.zeros((1, 2)), np.zeros((1,))
    if verbose:
        print(f"{len(uv)} points after projection")
    return uv, z


def get_projected_fusion_img(uvs, zs, img, radius=2):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    overlay = hsv_img.copy()

    # sort and draw
    uvs, zs = sort_points(uvs, zs)
    _min = np.min(zs)
    dist_norm = (zs - _min / 2) / (min(100, np.max(zs) - _min + 1e-10))  # 50m
    dist_norm = np.clip(dist_norm * 90, 0, 180)
    for i in range(uvs.shape[0]):
        cv2.circle(overlay, (int(uvs[i, 0]), int(uvs[i, 1])),
                   radius=radius,
                   color=(int(dist_norm[i]), 255, 255),
                   thickness=-1)

    overlay = cv2.cvtColor(overlay, cv2.COLOR_HSV2RGB)

    # Following line overlays transparent rectangle over the image
    image_new = fusion_images(overlay, img, alpha=0.7)
    return image_new


def show_projected_images(pc: np.ndarray, image: np.ndarray, intrinsic: np.ndarray, extrinsics: list,
                          names: list = None, radius=2, verbose=False):
    count = len(extrinsics)
    assert names is None or count == len(names)
    if names is None:
        names = [''] * count
    depth_images = [
        get_projected_fusion_img(*get_projected_pts(pc, intrinsic, image.shape,
                                                    extrinsic, verbose=verbose), img=image, radius=radius)
        for extrinsic in extrinsics
    ]

    base_size = 12
    fig, axes = plt.subplots(count, 1, figsize=(
        base_size * 3, base_size * count))
    for i, (name, depth_image) in enumerate(zip(names, depth_images)):
        ax = axes[i] if count > 1 else axes
        ax.set_title(name)
        ax.imshow(depth_image)
    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pcd_file', type=str)
    parser.add_argument('image_file', type=str)
    parser.add_argument('config_file', type=str)
    parser.add_argument('-c', '--camera', type=int, default=0)

    args = parser.parse_args()
    pc = load_pcd(args.pcd_file)
    img = Image.open(args.image_file)

    # config
    with open(args.config_file, encoding='utf8') as f:
        obj = json.load(f)
        config = obj[f'3d_img{args.camera}']
        fu, fv, cu, cv = [
            config['camera_internal'][key]
            for key in ['fx', 'fy', 'cx', 'cy']
        ]

        intrinsic = np.array([
            [fu, 0, cu, 0],
            [0, fv, cv, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        extrinsic = np.array(config["camera_external"]).reshape(4, 4)
        extrinsic = np.linalg.inv(extrinsic)
        image = np.asarray(img)
        if len(image.shape) == 2:
            image = image[..., None].repeat(3, axis=-1)
        show_projected_images(pc, image, intrinsic, [extrinsic])


if __name__ == '__main__':
    main()
