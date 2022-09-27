class CameraViewHandler {
    constructor(camera, view) {
        this.handler = new DragCamera(camera, view);
        view.addEventListener("mousemove", (e) => this.onMouseMove(e));
        view.addEventListener("mousedown", (e) => this.onMouseDown(e));
        view.addEventListener("mouseup", (e) => this.onMouseUp(e));
        view.addEventListener("mouseleave", (e) => this.onMouseLeave(e));
        view.addEventListener("wheel", (e) => this.onWheel(e));
    }

    changleHandler(handlerTag) {
        const camera = this.handler.camera;
        const view = this.handler.view;
        this.handler.dispose();
        if (handlerTag == "hover")
            this.handler = new HoverCamera(camera, view);
        else if (handlerTag == "drag")
            this.handler = new DragCamera(camera, view);
        else if (handlerTag == "sway")
            this.handler = new SwayCamera(camera, view);
        else if (handlerTag == "wonder")
            this.handler = new WonderCamera(camera, view);
        else
            console.log(`Handler Type ${handlerTag} not found`);
    }

    onWheel(e) { this.handler.onWheel(e) }
    onMouseMove(e) { this.handler.onMouseMove(e) }
    onMouseDown(e) { this.handler.onMouseDown(e) }
    onMouseUp(e) { this.handler.onMouseUp(e) }
    onMouseLeave(e) { this.handler.onMouseLeave(e) }
}

class CameraTypeBase {
    constructor(camera, view) {
        this.WHEEL_SPEED = .005;
        this.BASELINE = 0.45;
        this.LOOKAT_POINT = new THREE.Vector3(0, 0, -2);

        this.camera = camera;
        this.view = view;
        this.position = {"x": 0.5, "y": 0.5, "z": 0.0};
        this.update();
    }

    update() {
        const halfBaseline = 0.5 * this.BASELINE;
        this.camera.position.x = -halfBaseline * (2 * this.position.x - 1);
        this.camera.position.y = halfBaseline * (2 * this.position.y - 1);
        this.camera.position.z = this.position.z;
        this.camera.lookAt(this.LOOKAT_POINT);
    }

    onWheel(e) {
        this.position.z += this.WHEEL_SPEED * e.deltaY;
        e.preventDefault();
        this.update();
    }

    onMouseMove(e) {    }

    onMouseDown(e) {    }

    onMouseUp(e) {    }

    onMouseLeave(e) {    }

    dispose() { }
}

class HoverCamera extends CameraTypeBase {
    constructor(camera, view) {
        super(camera, view);
    }

    onMouseMove(e) {
        this.position.x = e.offsetX / this.view.clientWidth;
        this.position.y = e.offsetY / this.view.clientHeight;
        this.update();
    }
}



class DragCamera extends CameraTypeBase {
    constructor(camera, view) {
        super(camera, view);

        this.dragging = false;
        this.previous = {};
    }

    onMouseMove(e) {
        if (!this.dragging)
            return;
        this.position.x += (e.offsetX - this.previous.x) / this.view.clientWidth;
        this.position.y += (e.offsetY - this.previous.y) / this.view.clientHeight;
        this.previous = {"x": e.offsetX, "y": e.offsetY};
  
        this.update();
    }

    onMouseDown(e) {
        this.previous = {"x": e.offsetX, "y": e.offsetY};
        this.dragging = true;
    }

    onMouseUp(e) {
        this.dragging = false;
    }

    onMouseLeave(e) {
        this.onMouseUp(e);
    }
}

class SwayCamera extends CameraTypeBase {
    constructor(camera, view) {
        super(camera, view);
        this.delta_ms = 25;
        this.timer = setInterval(() => this.tick(), this.delta_ms);
        this.start_time = Date.now();
    }

    tick() {
        const elapsed = Date.now() - this.start_time;
        this.position.x = 0.5 + 0.5 * Math.sin(Math.PI * elapsed / 3000);
        this.update();
    }

    dispose() {
        clearInterval(this.timer);
    }
}

class WonderCamera extends CameraTypeBase {
    constructor(camera, view) {
        super(camera, view);
        this.delta_ms = 25;
        this.timer = setInterval(() => this.tick(), this.delta_ms);
        this.start_time = Date.now();
    }

    tick() {
        const elapsed = Date.now() - this.start_time;
        this.position.x = 0.5 + 0.5 * Math.cos(Math.PI / 2 + Math.PI * elapsed / 3000);
        this.position.y = 0.5 + 0.5 * Math.sin(Math.PI * elapsed / 4300);
        this.position.z = 0.1 * Math.sin(Math.PI * elapsed / 8012);

        this.update();
    }

    dispose() {
        clearInterval(this.timer);
    }
}