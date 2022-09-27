colormaps = {
  'magma': [
    0.001462, 0.000466, 0.013866, 1.0, 0.013708, 0.011771, 0.068667, 1.0,
    0.039608, 0.031090, 0.133515, 1.0, 0.074257, 0.052017, 0.202660, 1.0,
    0.113094, 0.065492, 0.276784, 1.0, 0.159018, 0.068354, 0.352688, 1.0,
    0.211718, 0.061992, 0.418647, 1.0, 0.265447, 0.060237, 0.461840, 1.0,
    0.316654, 0.071690, 0.485380, 1.0, 0.366012, 0.090314, 0.497960, 1.0,
    0.414709, 0.110431, 0.504662, 1.0, 0.463508, 0.129893, 0.507652, 1.0,
    0.512831, 0.148179, 0.507648, 1.0, 0.562866, 0.165368, 0.504692, 1.0,
    0.613617, 0.181811, 0.498536, 1.0, 0.664915, 0.198075, 0.488836, 1.0,
    0.716387, 0.214982, 0.475290, 1.0, 0.767398, 0.233705, 0.457755, 1.0,
    0.816914, 0.255895, 0.436461, 1.0, 0.863320, 0.283729, 0.412403, 1.0,
    0.904281, 0.319610, 0.388137, 1.0, 0.937221, 0.364929, 0.368567, 1.0,
    0.960949, 0.418323, 0.359630, 1.0, 0.976690, 0.476226, 0.364466, 1.0,
    0.986700, 0.535582, 0.382210, 1.0, 0.992785, 0.594891, 0.410283, 1.0,
    0.996096, 0.653659, 0.446213, 1.0, 0.997325, 0.711848, 0.488154, 1.0,
    0.996898, 0.769591, 0.534892, 1.0, 0.995131, 0.827052, 0.585701, 1.0,
    0.992440, 0.884330, 0.640099, 1.0, 0.987053, 0.991438, 0.749504, 1.0],
  'rainbow': [
    0.18995, 0.07176, 0.23217, 1.0, 0.22500, 0.16354, 0.45096, 1.0,
    0.25107, 0.25237, 0.63374, 1.0, 0.26816, 0.33825, 0.78050, 1.0,
    0.27628, 0.42118, 0.89123, 1.0, 0.27543, 0.50115, 0.96594, 1.0,
    0.25862, 0.57958, 0.99876, 1.0, 0.21382, 0.65886, 0.97959, 1.0,
    0.15844, 0.73551, 0.92305, 1.0, 0.11167, 0.80569, 0.84525, 1.0,
    0.09267, 0.86554, 0.76230, 1.0, 0.12014, 0.91193, 0.68660, 1.0,
    0.19659, 0.94901, 0.59466, 1.0, 0.30513, 0.97697, 0.48987, 1.0,
    0.42778, 0.99419, 0.38575, 1.0, 0.54658, 0.99907, 0.29581, 1.0,
    0.64362, 0.98999, 0.23356, 1.0, 0.72596, 0.96470, 0.20640, 1.0,
    0.80473, 0.92452, 0.20459, 1.0, 0.87530, 0.87267, 0.21555, 1.0,
    0.93301, 0.81236, 0.22667, 1.0, 0.97323, 0.74682, 0.22536, 1.0,
    0.99314, 0.67408, 0.20348, 1.0, 0.99593, 0.58703, 0.16899, 1.0,
    0.98360, 0.49291, 0.12849, 1.0, 0.95801, 0.39958, 0.08831, 1.0,
    0.92105, 0.31489, 0.05475, 1.0, 0.87422, 0.24526, 0.03297, 1.0,
    0.81608, 0.18462, 0.01809, 1.0, 0.74617, 0.13098, 0.00851, 1.0,
    0.66449, 0.08436, 0.00424, 1.0, 0.47960, 0.01583, 0.01055, 1.0],

}

function simpleTextureMaterial(texture) {
    const material = new THREE.ShaderMaterial({
        vertexShader: `
                        varying vec2 vTexCoords;
                        void main() {
                          vec4 pos = vec4(position, 1.0);
                          // Perform cv -> gl transform for LMIs and LMDIs.
                          vTexCoords = vec2(uv.x, 1.0 - uv.y);
                          pos.yz *= vec2(1.0);
                          gl_Position = projectionMatrix * modelViewMatrix * pos;
                        }`,
        fragmentShader: `
                        varying vec2 vTexCoords;
                        uniform sampler2D atlas;
                        uniform sampler2D atlas_a;

                        void main() {
                          gl_FragColor = texture2D(atlas, vTexCoords);
//                          gl_FragColor_a = texture2D(atlas_a, vTexCoords);
                          // TODO(rover): remove premult.
                          gl_FragColor.rgb *=  texture2D(atlas_a, vTexCoords).r;
                          gl_FragColor.a *= texture2D(atlas_a, vTexCoords).r;
                        }`,
        uniforms: { 'atlas': { 'value': texture[0] }, 'atlas_a': { 'value': texture[1] } },
    });
    return defaultMaterialProperties(material);
}

function simpleTextureMaterialWithStanaloneAlpha(texture) {
    const material = new THREE.ShaderMaterial({
        vertexShader: `
                        varying vec2 vTexCoords;
                        void main() {
                          vec4 pos = vec4(position, 1.0);
                          // Perform cv -> gl transform for LMIs and LMDIs.
                          vTexCoords = vec2(uv.x, 1.0 - uv.y);
                          pos.yz *= vec2(1.0);
                          gl_Position = projectionMatrix * modelViewMatrix * pos;
                        }`,
        fragmentShader: `
                        varying vec2 vTexCoords;
                        uniform sampler2D atlas;
                        uniform sampler2D atlas_a;

                        void main() {
                          gl_FragColor = texture2D(atlas, vTexCoords);
                          gl_FragColor_a = texture2D(atlas_a, vTexCoords);
                          // TODO(rover): remove premult.
                          gl_FragColor.rgb *= gl_FragColor_a.a;
                        }`,
        uniforms: { 'atlas': { 'value': texture[0] }, 'atlas_a': { 'value': texture[1] } },
    });
    return defaultMaterialProperties(material);
}

function depthMaterial(texture, colormapName='magma') {
  const colors = colormaps[colormapName];
  if (colors) {
    const rgba = new Uint8Array(colors.map(el => el * 255));
    const dataTex = new THREE.DataTexture( rgba, 32, 1, THREE.RGBAFormat );
    dataTex.minFilter = THREE.LinearFilter;
    dataTex.magFilter = THREE.LinearFilter;
    dataTex.needsUpdate = true;
    return depthMaterialWithColormap(texture, dataTex);
  }
  else
    console.log(`Color map ${colormapName} not found`)
}

function depthMaterialWithColormap(texture, colormapTexture) {
  const material = new THREE.ShaderMaterial({
    vertexShader: `
                    varying vec2 vTexCoords;
                    varying float viewZ;
                    void main() {
                      vec4 pos = vec4(position, 1.0);
                      // Perform cv -> gl transform for LMIs and LMDIs.
                      vTexCoords = vec2(uv.x, 1.0 - uv.y);
                      pos.yz *= vec2(1.0);
                      gl_Position = projectionMatrix * modelViewMatrix * pos;
                      viewZ = -(modelViewMatrix * pos).z * 1.2; // factor 1.2 to increase dynamic range
                      viewZ = 1.0 / viewZ; // disparity
                    }`,
    fragmentShader: `
                    varying vec2 vTexCoords;
                    varying float viewZ;
                    uniform sampler2D atlas;
                    uniform sampler2D colormap;

                    void main() {
                      vec4 texture_color = texture2D(atlas, vTexCoords);
                      vec4 shading_color = texture2D(colormap, vec2(viewZ, 0.5));
                      gl_FragColor = vec4(shading_color.x, shading_color.y, shading_color.z, texture_color.r);
                      // gl_FragColor = vec4( viewZ, viewZ, viewZ, texture_color.a);

                      gl_FragColor.rgb *= gl_FragColor.a;
                    }`,
    uniforms: { 'atlas': { 'value': texture[1] }, 'colormap': { 'value': colormapTexture } },
  });
  
  return defaultMaterialProperties(material);
}

function defaultMaterialProperties(material) {
    material.blending = THREE.CustomBlending;
    material.blendEquation = THREE.AddEquation;
    material.blendSrc = THREE.OneFactor;
    material.blendDst = THREE.OneMinusSrcAlphaFactor;
    return material;
}