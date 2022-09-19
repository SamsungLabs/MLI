
Dataset setup:

1) Download all ten parts of sword.zip (sword.zip.partab ... sword.zip.partak).
2) Run ```zip -F sword.zip --out sword.zip```
3) Unpack sword.zip

Dataset structure:

| root/:
|---- dataset.csv
|---- videos/
    |---- scene_id1
        |---- frame1.jpg
        |---- frame2.jpg
    |---- scene_id2
        |---- frame1.jpg
        |---- frame2.jpg
|---- views/
    |---- scene_id1.txt
    |---- scene_id2.txt


dataset.csv fields:
    General:
        scene_id - scene id

    File paths:
        images_path - scene images path in full resolution
        images_path_x - images path in x resolution in short side, for example:
            images_path_256 - it's images with 455x256 resolution if source was in full HD (1920x1080)
        colmap_path - path to colmap scene data (points3D.bin, images.bin, cameras.bin)
        views_path - path to txt file with scene RealEstate10k like views params (intrinsics, extrinsics).

    Scene params:
        num_points - number of 3d points in colmap scene
        num_views - number of views
        p_x - depth for x percentile, calculated via all scene views
        error_mean - mean of colmap 3d point error
        error_std - std of colmap 3d point error

    Intrinsic (relative):
        f_x - focal distance x
        f_y - focal distance y
        c_x - central point x
        c_y - central point y
        original_resolution_x - original image x resolution (which used for colmap)
        original_resolution_y - original image y resolution (which used for colmap)
