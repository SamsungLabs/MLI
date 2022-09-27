async function tf_gen_planes(height, width, align_corners=true) {
    const verts = tf.tidy(() => {
        const [delta_x, delta_y] = align_corners ? [0, 0] : [0.5 / width, 0.5 / height]
        
        const xs = tf.linspace(delta_x, 1 - delta_x, width)
        const ys = tf.linspace(delta_y, 1 - delta_y, height)
        const x_verts = tf.broadcastTo(xs, [height, width])
        const y_verts = tf.broadcastTo(ys.reshape([height, 1]), [height, width])
        return tf.stack([x_verts, y_verts], axis=2).reshape([-1, 2])
    });
        
    let faces_base_vertices_ids = tf.range(0, width * (height - 1), 1)
    const valid = tf.notEqual(faces_base_vertices_ids.mod(width), width-1)
    faces_base_vertices_ids = await tf.booleanMaskAsync(faces_base_vertices_ids, valid)
    valid.dispose()
    const [faces, verts_uvs] = tf.tidy(() => {
        const quade_faces = tf.tensor([[0, width, 1 + width], [0, 1 + width, 1]])
        const f = quade_faces.reshape([1, 2, 3]).add(faces_base_vertices_ids.reshape([-1, 1, 1])).reshape([-1, 3])
        faces_base_vertices_ids.dispose()

        const flip_v = verts.slice([0, 1], [-1, 1]).mul(-1).add(1)
        const verts_uvs = tf.concat([verts.slice([0, 0], [-1, 1]), flip_v], axis=1)

        return [f, verts_uvs]
    });

    return [verts, faces, verts_uvs]
}

async function read_jpg_depth(url, meta_data) {
    const [low, high] = meta_data
    const response = await fetch(url)
    const imageBlob = await response.blob()
    const image = await blob_to_image(imageBlob)
    return tf.tidy(() => {
        const tensor = tf.browser.fromPixels(image)
        const quant_depth = tensor.slice([0, 0, 0], [-1, -1, 1]);  // All channels are equal, take just the first
        // Rescale depth
        const rescaled = quant_depth.mul((high - low) / 255.0).add(low)
        return rescaled;
    });
}

async function read_txt_depth(url) {
    const depth_str = await (await fetch(url)).text()
    const as_list = JSON.parse(depth_str)
    const tensor = tf.tensor(as_list)
    return tensor
}

function blob_to_image(imageBlob) {
    const blobUrl = URL.createObjectURL(imageBlob)
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = err => reject(err);
        img.src = blobUrl;
    });
}

class ProjectiveCamera {
    constructor(extrinsics, intrinsics, im_height, im_width) {
        this.R = extrinsics.slice([0, 0], [3, 3]);
        this.T = extrinsics.slice([0, 3], [3, 1]);
        this.intrinsics = intrinsics;
        this.inv_K = this.inverse_pinhole(this.intrinsics)
        this.im_height = im_height
        this.im_width = im_width
    }

    static async from_csv(csv_path = '3/scene.csv') {
        const response = await fetch(csv_path);
        const csv_str = await response.text();
        const parsed = Papa.parse(csv_str, { header: true, dynamicTyping: true });
        const camera_line = parsed.data[0]
        const extrinsics = JSON.parse(camera_line['extrinsic_re'])
        const intrinsics = JSON.parse(camera_line['intrinsics_re'])
        const size = JSON.parse(camera_line['frame_size'])
        return ProjectiveCamera.from_params(extrinsics, intrinsics, size)
    }

    static async from_meta(meta_data) {
        const extrinsics = meta_data['extrinsic_re']
        const intrinsics = meta_data['intrinsics_re']
        const size = meta_data['frame_size']
        return ProjectiveCamera.from_params(extrinsics, intrinsics, size)
    }

    static from_params(extrinsics, intrinsics, size, use_identity_e=true) {
        let ext_tf = tf.tensor(extrinsics).reshape([3, 4]);
        if (use_identity_e)
            ext_tf = tf.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]);
        const int_tf = tf.tensor(intrinsics).reshape([3, 3]);
        return new ProjectiveCamera(ext_tf, int_tf, size[0], size[1]);
    }

    inverse_pinhole(intrinsics) {
        const data = intrinsics.dataSync()
        const fx = data[0]
        const cx = data[2]
        const fy = data[4]
        const cy = data[5]
        return tf.tensor([[1/fx, 0, -cx/fx], [0, 1/fy, -cy/fy], [0, 0, 1]])
    }

    pixel_to_world(pixels, depths) {
        // pixels shape B x 2
        // depths shape B x 1
        return tf.tidy(() => {
            const points = tf.concat([pixels, tf.onesLike(depths)], axis=1).mul(depths)
            return this.film_to_world(points);
        });
    }

    film_to_world(points) {
        // points shape B x 3
        return tf.tidy(() => {
            const camera_points = this.inv_K.matMul(points.transpose())  // 3 x B
            const world_points = this.R.transpose().matMul(camera_points.sub(this.T)) // 3 x B
            return world_points.transpose()
        });
    }
}
