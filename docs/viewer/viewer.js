//const baseUrl = "scenes/";
//const num_layers = 4;
//const layersIds = [...Array(num_layers).keys()].map(i => String(i).padStart(2, '0'))
const depth_scale = 4;
const stats = new Stats();

function loadAllTextures(baseUrl, layersIds) {
  const textureLoader = new THREE.TextureLoader()
  return layersIds.map(id => {
    texAtlasUrl = `${baseUrl}/layer_${id}.jpg`
    const texture = textureLoader.load(texAtlasUrl);
    texture.generateMipmaps = false;
    texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.minFilter = THREE.LinearFilter;

    texAtlasAUrl = `${baseUrl}/layer_alpha_${id}.jpg`
    const texture_a = textureLoader.load(texAtlasAUrl);
    texture_a.generateMipmaps = false;
    texture_a.wrapS = texture_a.wrapT = THREE.ClampToEdgeWrapping;
    texture_a.minFilter = THREE.LinearFilter;

    return [texture, texture_a];
  });
}



async function loadAllDepths(meta_data, baseUrl, layersIds, extension='jpg') {
  let loader = null
  if (['png', 'jpg'].includes(extension)) {
    loader = (depth_path, meta) => read_jpg_depth(depth_path, meta)
  }
  else if (extension == "txt") {
    loader = (depth_path, meta) => read_txt_depth(depth_path)
  }
  const tags = layersIds.map(id => `layer_depth_${id}`)
  const depth_paths = tags.map(tag => `${baseUrl}/${tag}.${extension}`)
  return await Promise.all(depth_paths.map((url, index) => loader(url, meta_data[tags[index]])))
}

function setupCamera(canvas) {
  const camera = new THREE.PerspectiveCamera(72, canvas.clientWidth / canvas.clientHeight, .1, 1000);
  const view = document.getElementById('scene-viewer');

  return [camera, new CameraViewHandler(camera, view)];
}

function setupNeutralHoverCamera(canvas, idScene='scene-viewer') {
  const camera = new THREE.PerspectiveCamera(
      72, canvas.clientWidth / canvas.clientHeight, .1, 10000);

  const view = document.getElementById(idScene);
  const BASELINE = 0.3;
  view.addEventListener('wheel', e => {
    const WHEEL_SPEED = .005;
    camera.position.z += WHEEL_SPEED * e.deltaY;
    camera.position.clampLength(0, 0.5 * BASELINE);
    e.preventDefault();
  });

  const LOOKAT_DISTANCE = 1;
  const LOOKAT_POINT = new THREE.Vector3(0, 0, -LOOKAT_DISTANCE);
  view.addEventListener('mousemove', e => {
    const halfBaseline = 0.5 * BASELINE;
    const x = e.offsetX / view.clientWidth;
    const y = e.offsetY / view.clientHeight;
    camera.position.x = -halfBaseline * (2 * x - 1);
    camera.position.y = halfBaseline * (2 * y - 1);
    camera.position.clampLength(0.0, halfBaseline);
    camera.lookAt(LOOKAT_POINT);
  });
  return camera;
}

function setupRenderer(canvas, camera, scene) {
  const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
//  renderer.setClearColor(0x000000, 0);
  renderer.setClearColor( 0xffffff, 0);
//  renderer.setClearColor( 0x808080, 0);
//  scene.background = new THREE.Color( 0x808080 );
  scene.background = new THREE.Color( 0xffffff );
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setAnimationLoop(() => {
    renderer.render(scene, camera);
    stats.update();
  });
}

function buildGeometry(verts, faces_flat, uv_flat, depths, ref_camera) {
  // Depth must be reversed to match the uv parametrization
  depths = tf.image.resizeBilinear(depths, [ref_camera.im_height, ref_camera.im_width])
  const rev_depth = tf.reverse(depths.reshape([ref_camera.im_height, ref_camera.im_width]), axis=0).reshape([-1, 1])
  const v3 = ref_camera.pixel_to_world(verts.reshape([-1, 2]), rev_depth).reshape([ref_camera.im_height, ref_camera.im_width, 3])
  const positions = v3.flatten().dataSync()
  const geometry = new THREE.BufferGeometry();
 
  geometry.setIndex(Array.from(faces_flat));
  geometry.setAttribute('position', new THREE.BufferAttribute( positions, 3 ) );
  geometry.setAttribute('uv', new THREE.BufferAttribute(uv_flat, 2));

  return geometry;
}

function buildMesh(geometry, material) {
  const mesh = new THREE.Mesh( geometry, material );
  mesh.frustumCulled = false;
  mesh.scale.set(1, 1, -0.7);
  mesh.position.z = 0;
  return mesh
}

function buildScene(geometries, materials) {
  const scene = new THREE.Scene();
  for (let i = 0; i < geometries.length; i++) {
    const mesh = buildMesh(geometries[i], materials[i])
    scene.add(mesh)
  }
  return scene
}

async function startDisplay(inputPath, num_layers=4) {
  const canvas = document.getElementById('viewer-canvas');
  const layersIds = [...Array(num_layers).keys()].map(i => String(i).padStart(2, '0'))

//   document.getElementById('viewer-path');
  const baseUrl = inputPath
//  const baseUrl = document.getElementById('base-path').value;
  const meta_data = await fetch(`${baseUrl}/meta.json`).then(result => result.json())
  const ref_camera = await ProjectiveCamera.from_meta(meta_data)
  ref_camera.im_height = ref_camera.im_height / depth_scale
  ref_camera.im_width = ref_camera.im_width / depth_scale
  const depths = await loadAllDepths(meta_data, baseUrl, layersIds)
  const [verts, faces, uv] = await tf_gen_planes(ref_camera.im_height, ref_camera.im_width)
  const faces_flat = faces.toInt().flatten().dataSync()
  const uv_flat = uv.flatten().dataSync()

  const geometries = depths.map(depth => buildGeometry(verts, faces_flat, uv_flat, depth, ref_camera))
  const textures_loaded = loadAllTextures(baseUrl, layersIds)
  const materials = textures_loaded.map(simpleTextureMaterial)
  const scene = buildScene(geometries, materials);

  const [camera, handler] = setupCamera(canvas);
  
  setCameraControlEvents(handler);
  setViewModeEvents(scene.children, textures_loaded);
  setStatsDsiplay();
  setupRenderer(canvas, camera, scene);
}


async function startMainDisplay(inputPath, canvasName, idScene='scene-viewer') {
  const canvas = document.getElementById(canvasName);
  const baseUrl = inputPath
  const num_layers = 4;
  const layersIds = [...Array(num_layers).keys()].map(i => String(i).padStart(2, '0'))
//  const baseUrl = document.getElementById('base-path').value;
  const meta_data = await fetch(`${baseUrl}/meta.json`).then(result => result.json())
  const ref_camera = await ProjectiveCamera.from_meta(meta_data)
  const depths = await loadAllDepths(meta_data, baseUrl, layersIds)
  const [verts, faces, uv] = await tf_gen_planes(ref_camera.im_height, ref_camera.im_width)
  const faces_flat = faces.toInt().flatten().dataSync()
  const uv_flat = uv.flatten().dataSync()

  const geometries = depths.map(depth => buildGeometry(verts, faces_flat, uv_flat, depth, ref_camera))
  const materials = loadAllTextures(baseUrl, layersIds).map(simpleTextureMaterial)
  const scene = buildScene(geometries, materials);

  const camera = setupNeutralHoverCamera(canvas, idScene);
//  handler.changleHandler('hover')
//  setStatsDsiplay();
  setupRenderer(canvas, camera, scene);
}


async function startMainDisplayPlain(inputPath, canvasName, idScene='scene-viewer') {
  const canvas = document.getElementById(canvasName);
  const baseUrl = inputPath
  const num_layers = 4;
  const layersIds = [...Array(num_layers).keys()].map(i => String(i).padStart(2, '0'))
//  const baseUrl = document.getElementById('base-path').value;
  const meta_data = await fetch(`${baseUrl}/meta.json`).then(result => result.json())
  const ref_camera = await ProjectiveCamera.from_meta(meta_data)
  const depths = await loadAllDepths(meta_data, baseUrl, layersIds)
  const [verts, faces, uv] = await tf_gen_planes(ref_camera.im_height, ref_camera.im_width)
  const faces_flat = faces.toInt().flatten().dataSync()
  const uv_flat = uv.flatten().dataSync()

  const geometries = depths.map(depth => buildGeometry(verts, faces_flat, uv_flat, depth, ref_camera))
  const materials = loadAllTextures(baseUrl, layersIds).map(simpleTextureMaterial)
  const scene = buildScene(geometries, materials);
//  const camera = setupCamera(canvas)
  const [camera, handler] = setupCamera(canvas);
  handler.changleHandler("wonder")
//  const view = document.getElementById('scene-viewer');
//  camera = new WonderCamera(camera, view)
//  const camera = setupNeutralHoverCamera(canvas, idScene);
  setupRenderer(canvas, camera, scene);
}

function setStatsDsiplay() {
  panels = stats.dom.children;
  for (canvas of panels) {
    canvas.style.width = "120px";
    canvas.style.height = "72px";
  }
  document.body.appendChild(stats.dom);
}

function setCameraControlEvents(handler) {
  document.getElementById("hoverButton").addEventListener("click", (e) => handler.changleHandler("hover"));
  document.getElementById("dragButton").addEventListener("click", (e) => handler.changleHandler("drag"));
  document.getElementById("swayButton").addEventListener("click", (e) => handler.changleHandler("sway"));
  document.getElementById("wonderButton").addEventListener("click", (e) => handler.changleHandler("wonder"));
}

function setViewModeEvents(meshes, textures_loaded) {
  function setNewMode(mode) {
//    textures = meshes.map(mesh => mesh.material.uniforms.atlas.value)
    let materials = []
    if (mode == "normal")
      materials = textures_loaded.map((el) => simpleTextureMaterial(el));
    else
      materials = textures_loaded.map((el) => depthMaterial(el, colormapName=mode));
    for (i = 0; i < materials.length; i++)
      meshes[i].material = materials[i];
  }
  document.getElementById("normalViewButton").addEventListener("click", (e) => setNewMode("normal"));
  document.getElementById("rainbowViewButton").addEventListener("click", (e) => setNewMode("rainbow"));
  document.getElementById("magmaViewButton").addEventListener("click", (e) => setNewMode("magma"));
}

//document.querySelector("div[display-path]")
//const baseUrl = document.getElementById('base-path').value;
//document.getElementById("display").innerHTML = startDisplay();
//document.getElementById("display-main").innerHTML = startMainDisplay();

