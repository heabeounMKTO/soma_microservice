SCRFD_MODEL := models/det_10g.onnx
ARCFACE_MODEL := models/arcfaceresnet100-8.onnx
YOLOFACE_MODEL := models/yoloface_8n.onnx
LOCAL_CONTAINER_NAME := localhost:5000/face_api:latest
LOCAL_SERVER_PORT := 9999


local_container: 
	if [ -d "models/" ]; then echo "model folder found !"; else make get_models ; fi
	docker buildx build --platform linux/amd64 -t ${LOCAL_CONTAINER_NAME} .

run_local: local_container
	echo "running local server , through host port $(LOCAL_SERVER_PORT)"
	docker run -it -p ${LOCAL_SERVER_PORT}:9999 ${LOCAL_CONTAINER_NAME} 

# TODO build onnxruntime from source with musl
container_musl:
	docker buildx build --platform linux/amd64 -f Dockerfile.alpine -t localhost:5000/face_api_musl:latest .

get_models:
	mkdir -p venv
	mkdir -p models
	python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt && python3 get_models.py
